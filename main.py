from pathlib import Path
import numpy as np, os, cv2
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch_geometric.nn import PointNetConv
import torch.nn.functional as F
from yolox.models.yolo_head import YOLOXHead
from pytorch_lightning import Trainer
from yolox.utils import postprocess
import torchmetrics
import matplotlib.pyplot as plt
# import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

KITTI_DATASET_PATH = r"/home/tymek/Desktop/kitti_dataset"
BATCH_SIZE = 1
NUM_WORKERS = 0
INTERNAL_IMG_SIZE = 240, 1216


def visualize_lidar(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        cmap='viridis', s=0.5
    )

    # Label axes
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')

    # Optional: set equal aspect ratio
    ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])

    plt.show()

def read_kitti_calib(file_path: str):
    calib = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, value = line.split(":", 1)
            arr = np.array([float(x) for x in value.strip().split()])

            if key[0] == "P":
                calib[key] = arr.reshape((3, 4))
            else:
                matrix = np.zeros((4, 4), dtype=np.float32)
                matrix[3, 3] = 1.0
                
                if arr.shape == (12,):
                    matrix[:3, :] = arr.reshape((3, 4))
                elif arr.shape == (9,):
                    matrix[:3, :3] = arr.reshape((3, 3))
                
                calib[key] = matrix
            
    return calib

def read_kitti_dataset(dataset_path: str, filename: str):
    SUPPORTED_CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    calib = read_kitti_calib(os.path.join(dataset_path, "calib", f"{filename}.txt"))
    image = cv2.imread(os.path.join(dataset_path, "image_2", f"{filename}.png"))
    height, width, _ = image.shape
    img_shape = (height, width)

    lidar_data = np.fromfile(os.path.join(dataset_path, "velodyne", f"{filename}.bin"), dtype=np.float32)
    lidar_data = lidar_data.reshape(-1, 4)
    # visualize_lidar(lidar_data[:, :3])
    lidar_data = lidar_data[lidar_data[:, 0] >= 0]
    # visualize_lidar(lidar_data[:, :3])

    multiplier = calib["P2"] @ calib["R0_rect"] @ calib["Tr_velo_to_cam"]
    
    projected_points = (multiplier @ np.vstack([lidar_data[:, :3].T, np.ones((1, lidar_data.shape[0]))])).T
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]
    projected_points = projected_points[:, :-1]

    mask_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)

    visible_points_img = projected_points[mask_points]
    visible_points_lidar = lidar_data[mask_points]

    # print(visible_points_img.shape)
    # print(visible_points_lidar.shape)
    # visualize_lidar(visible_points_lidar)

    pos = visible_points_lidar[:, :3]
    intensity = visible_points_lidar[:, -1]

    # for i in range(visible_points_img.shape[0]):
    #     cv2.circle(image, (int(visible_points_img[i, 0]), int(visible_points_img[i, 1])), 0, (0,0,255), -1)
    
    # cv2.imshow("Lidar", image)
    # cv2.waitKey(0)

    labels_full = []
    with open(os.path.join(dataset_path, "label_2", f"{filename}.txt")) as f:
        while True:
            line = f.readline()
            if line == "":
                break
            values = line.split()
            class_id = values[0]
            if class_id not in SUPPORTED_CLASSES.keys():
                continue
            bbox = np.array([float(val) for val in values[4:8]], dtype=np.float32)
            score = np.float32(values[14])
            label = (SUPPORTED_CLASSES[class_id], score)

            labels_full.append((bbox, label))
    
    labels_full.sort(key=lambda val: val[1][1], reverse=True)
    
    bboxes = np.array([label[0] for label in labels_full])
    labels = np.array([label[1][0] for label in labels_full]).reshape(-1, 1)
    
    return intensity, pos, bboxes, labels, calib, img_shape

def create_graph(pos: torch.Tensor, radius: float, k: int):
    N = pos.size(0)
    device = pos.device

    # ---- pairwise squared distance matrix ----
    pos2 = (pos ** 2).sum(dim=1, keepdim=True)
    dist2 = pos2 + pos2.t() - 2.0 * pos @ pos.t()

    # ---- radius mask (exclude self for now) ----
    mask = (dist2 <= radius ** 2) & (dist2 > 0)

    # ---- KNN filtering inside radius ----
    if k is not None:
        # set distances outside radius to +inf
        dist2_masked = dist2.clone()
        dist2_masked[~mask] = float("inf")

        # take k nearest neighbors
        knn_dist, knn_idx = torch.topk(
            dist2_masked, k, largest=False, dim=1
        )

        knn_mask = torch.zeros_like(mask)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx)
        valid = torch.isfinite(knn_dist)
        print("valid", valid.size())
        print("knn_mask", knn_mask.size())

        knn_mask[row_idx[valid], knn_idx[valid]] = True
        mask = mask & knn_mask

    # ---- self-loops ----
    idx = torch.arange(N, device=device)
    mask[idx, idx] = True

    # ---- edge index ----
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    return edge_index

class KITTIGraphDataset(Dataset):
    def __init__(self, dir, split_range):
        self.dataset_dir = dir
        filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.dataset_dir, "image_2"))])
        bounds = [int(round(split_range[0]*len(filenames))), int(round(split_range[1]*len(filenames)))]
        self.filenames = [val for i, val in enumerate(filenames) if i >= bounds[0] and i < bounds[1]]
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        print(self.filenames[idx])
        intensity, pos, bboxes, labels, calib, img_shape = read_kitti_dataset(self.dataset_dir, self.filenames[idx])
        print("read_kitti_dataset return shapes:")
        print("intensity.shape", intensity.shape)
        print("pos.shape", pos.shape)
        print("bboxes.shape", bboxes.shape)
        print("labels.shape", labels.shape)

                                                                        # N - number of points, D - number of detections per frame
        x_torch = torch.tensor(intensity).unsqueeze(1)                  # (N, 1)
        pos_torch = torch.from_numpy(pos).contiguous()                  # (N, 3)
        edge_index_torch = create_graph(pos_torch, radius=0.2, k=15)    # (2, ...)
        bboxes_torch = torch.tensor(bboxes)                             # (D, 4)
        labels_torch = torch.tensor(labels)                             # (D,)

        print("edge_index.shape", edge_index_torch.size())

        return {
            "x": x_torch.to(torch.float32),
            "pos": pos_torch.to(torch.float32),
            "edge_index": edge_index_torch.to(torch.long),
            "bboxes": bboxes_torch.to(torch.float32),
            "labels": labels_torch.to(torch.long),
            "calib" : calib,
            "img_shape": img_shape
        }

def build_yolox_targets(batches):
    print("Building yolo targets")

    def create_targets_from_batch(id, batch):
        return torch.hstack([torch.full((len(batch["labels"]), 1), id), batch["labels"], batch["bboxes"]])

    return torch.vstack([create_targets_from_batch(id, batch) for id, batch in enumerate(batches)])
        
        
def lidar_collate_fn(batches):
    print("lidar_collate_fn")

    x = torch.cat([batch["x"] for batch in batches], dim=0)
    pos = torch.cat([batch["pos"] for batch in batches], dim=0)

    edge_index, offset = [], 0
    for batch in batches:
        edge_index.append(batch["edge_index"] + offset)
        offset += batch["x"].size(0)
    edge_index = torch.cat(edge_index, dim=1)

    batch_vec = torch.cat([
        torch.full((batch["x"].size(0),), i, dtype=torch.long)
        for i, batch in enumerate(batches)
    ])

    targets = build_yolox_targets(batches)
    print("YOLO targets.size:", targets.size())

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "batch": batch_vec,
        "bboxes": [batch["bboxes"] for batch in batches],
        "labels": [batch["labels"] for batch in batches],
        "targets": targets,
        "calib": [batch["calib"] for batch in batches],
        "img_shape": [batch["img_shape"] for batch in batches]
    }

class KITTIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        split_train = 0.7
        split_val = 0.15
        split_test = 0.15
        
        self.train_dataset = KITTIGraphDataset(KITTI_DATASET_PATH, [0.0, split_train])
        self.val_dataset = KITTIGraphDataset(KITTI_DATASET_PATH, [split_train, split_train + split_val])
        self.test_dataset = KITTIGraphDataset(KITTI_DATASET_PATH, [split_train + split_val, 1.0])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lidar_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lidar_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lidar_collate_fn)

    
def points_to_image(x, pos, batch, calib, img_shape):
    points_offset = (img_shape[0][1] - INTERNAL_IMG_SIZE[1], img_shape[0][0] - INTERNAL_IMG_SIZE[0])

    projected_points = (calib[0]["P2"] @ calib[0]["R0_rect"] @ calib[0]["Tr_velo_to_cam"] @ np.vstack([pos.detach().cpu().numpy().T, np.ones((1, pos.size(0)))])).T
    
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]
    projected_points = projected_points[:, :-1]
    projected_points = np.astype(projected_points, np.int16)
    projected_points -= np.array(points_offset).reshape((1, -1))

    mask = (projected_points[:, 0] >= 0) & (projected_points[:, 1] >= 0)
    projected_points = projected_points[mask]
    x = x[mask]

    img = np.zeros((batch.max()+1, x.size(1), INTERNAL_IMG_SIZE[1], INTERNAL_IMG_SIZE[0]))
    img[0, :, projected_points[:,0], projected_points[:, 1]] += x[:].detach().cpu().float().numpy() # TODO Adjust when batch > 1

    img_torch = torch.from_numpy(img).float().to(x.device)  # (B, C, W, H)

    pooled = F.max_pool2d(
        img_torch,
        kernel_size=4,
        stride=4
    )

    return pooled

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

class GraphBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = PointNetConv(local_nn=MLP(in_channels + 3, 64))
        self.conv2 = PointNetConv(local_nn=MLP(64 + 3, 128))
        self.conv3 = PointNetConv(local_nn=MLP(128 + 3, 256))

    def forward(self, x, pos, edge_index):
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)
        x = self.conv3(x, pos, edge_index)
        return x
            
    
class LidarYOLOX(nn.Module):
    def __init__(self, in_channels, num_classes, K):
        super().__init__()
        self.K = K
        self.backbone = GraphBackbone(in_channels)
        self.yolo_head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[256], strides=[4])

    def forward(self, x, pos, edge_index, batch, calib, img_shape, targets=None):
        point_feat = self.backbone(x, pos, edge_index)
        feat_maps = points_to_image(x=point_feat, pos=pos, batch=batch, calib=calib, img_shape=img_shape)

        return self.yolo_head([feat_maps], targets, imgs=None)

    
class LidarYOLOXModule(pl.LightningModule):
    def __init__(self, num_classes, in_channels, K, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.K = K
        self.model = LidarYOLOX(in_channels, num_classes, K)
        self.map_metric = torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def step(self, batch):
        print("Batch inside step function")
        print("intensity.shape", batch["x"].size())
        print("pos.shape", batch["pos"].size())
        print("edge_index.shape", batch["edge_index"].size())
        print("number of set of bboxes", len(batch["bboxes"]))
        print("number of labels", len(batch["labels"]))

        device = batch["x"].device
        
        loss_dict = self.model(
            batch["x"],
            batch["pos"],
            batch["edge_index"],
            batch["batch"],
            batch["calib"],
            batch["img_shape"],
            targets=batch["tragets"]
        )

        return loss_dict
    

    def training_step(self, batch, batch_idx):
        loss_dict = self.step(batch)
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = loss_dict
        
        self.log_dict(
            {
                "train/loss": loss,
                "train/iou_loss": iou_loss,
                "train/conf_loss": conf_loss,
                "train/cls_loss": cls_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        device = batch["x"].device
        outputs = self.model(
            batch["x"],
            batch["pos"],
            batch["edge_index"],
            batch["batch"],
            batch["calib"],
            batch["img_shape"],
            targets=None
        )
        
        preds = postprocess(
            outputs,
            num_classes=self.hparams.num_classes,
            conf_thre=0.5,
            nms_thre=0.5
        )

        pred_list = []
        target_list = []

        for i, pred in enumerate(preds):
            #print(batch["bboxes"][0][:5])
            #print(pred[:, :4][:5])
        
            if pred is None:
                pred_list.append({
                    "boxes": torch.zeros((0,4), device=device),
                    "scores": torch.zeros((0), device=device),
                    "labels": torch.zeros((0), dtype=torch.long, device=device),
                })
            else:
                pred_list.append({
                    "boxes": pred[:, :4],
                    "scores": pred[:, 4],
                    "labels": pred[:, 6].long(),
                })

            target_list.append({
                "boxes": batch["bboxes"][i].to(device),
                "labels": batch["labels"][i].to(device),
            })

        self.map_metric.update(pred_list, target_list)
    
    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP50", metrics["map_50"])
        self.log("val/mAP75", metrics["map_75"])
        self.map_metric.reset()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
    
    
def main():

    # wandb.login()
    # wandb_logger = WandbLogger(project="GNN_LiDAR",
    #                         entity="deep-neural-network-course",
    #                         group="gnn",
    #                         name="Mateusz Cierpik",
    #                         log_model=True)

    dm = KITTIDataModule(KITTI_DATASET_PATH, BATCH_SIZE)
    dm.setup()
    
    model = LidarYOLOXModule(num_classes=3, in_channels=1, K=K)
    trainer = Trainer(accelerator="cpu", devices=1, precision="16-mixed", max_epochs=10, log_every_n_steps=1)
    trainer.fit(model, datamodule=dm)
    
    # wandb.finish()
    # dm = KITTIDataModule(KITTI_DATASET_PATH, 3)
    # dm.setup()


if __name__ == "__main__":
    main()