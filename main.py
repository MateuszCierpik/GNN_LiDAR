from pathlib import Path
import numpy as np
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
import wandb
from lightning.pytorch.loggers import WandbLogger

class LidarGraphDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("*.pt"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)

        return {
            "x": data.x.float(),
            "pos": data.pos.float(),
            "edge_index": data.edge_index.long(),
            "bboxes": data.bboxes.float(),
            "labels": data.labels.long()
        }
        
        
def lidar_collate_fn(batch):
    x = torch.cat([b["x"] for b in batch], dim=0)
    pos = torch.cat([b["pos"] for b in batch], dim=0)

    edge_index, offset = [], 0
    for b in batch:
        edge_index.append(b["edge_index"] + offset)
        offset += b["x"].size(0)
    edge_index = torch.cat(edge_index, dim=1)

    batch_vec = torch.cat([
        torch.full((b["x"].size(0),), i, dtype=torch.long)
        for i, b in enumerate(batch)
    ])

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "batch": batch_vec,
        "bboxes": [b["bboxes"] for b in batch],
        "labels": [b["labels"] for b in batch]
    }
 
 
def xywh_to_xyxy(b):
    x, y, w, h = b.unbind(-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_cxcywh(b):
    x, y, w, h = b.unbind(-1)
    cx = x + w / 2
    cy = y + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def points_to_image(x, pos, batch, K, W=640, H=480, stride=8):
    X, Y, Z = pos.T

    u = (K[0,0] * X / Z + K[0,2])
    v = (K[1,1] * Y / Z + K[1,2])

    u = (u / stride).long()
    v = (v / stride).long()

    Wf = W // stride
    Hf = H // stride

    mask = (u >= 0) & (u < Wf) & (v >= 0) & (v < Hf)

    B = int(batch.max()) + 1
    C = x.size(1)

    feat_map = torch.zeros(B, C, Hf, Wf, device=x.device)

    for i in mask.nonzero().squeeze(1):
        feat_map[batch[i], :, v[i], u[i]] += x[i]

    return feat_map


def build_yolox_targets(batch, device):
    targets = []
    for b, l in zip(batch["bboxes"], batch["labels"]):
        if b.numel() == 0:
            targets.append(torch.zeros((0, 5), device=device))
        else:
            b = b.to(device)
            l = l.to(device)
            bb = xywh_to_cxcywh(b)
            t = torch.cat([l[:, None].float(), bb], dim=1)
            targets.append(t)

    max_n = max(t.size(0) for t in targets)
    out = torch.zeros(len(targets), max_n, 5, device=device)

    for i, t in enumerate(targets):
        out[i, :t.size(0)] = t

    return out


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
        self.yolo_head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[256], strides=[8])

    def forward(self, x, pos, edge_index, batch, targets=None):
        point_feat = self.backbone(x, pos, edge_index)
        feat_maps = points_to_image(x=point_feat, pos=pos, batch=batch, K=self.K)
        return self.yolo_head([feat_maps], targets, imgs=None)

    
class LidarYOLOXModule(pl.LightningModule):
    def __init__(self, num_classes, in_channels, K, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.K = K
        self.model = LidarYOLOX(in_channels, num_classes, K)
        self.map_metric = torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def step(self, batch):
        device = batch["x"].device
        targets = build_yolox_targets(batch, device)
        #print("targets[0][:5]:", targets[0][:5])
        
        loss_dict = self.model(
            batch["x"],
            batch["pos"],
            batch["edge_index"],
            batch["batch"],
            targets=targets
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
            targets=None
        )
        
        preds = postprocess(
            outputs,
            num_classes=self.hparams.num_classes,
            conf_thre=0.05,
            nms_thre=0.3
        )
        
        #print("GT:", xywh_to_xyxy(batch["bboxes"][0])[:3])
        #print("PRED:", preds[0][:3, :4])

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
                
            gt_xyxy = xywh_to_xyxy(batch["bboxes"][i].to(device))

            target_list.append({
                "boxes": gt_xyxy,
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
    wandb.login()
    wandb_logger = WandbLogger(project="GNN_LiDAR",
                            entity="deep-neural-network-course",
                            group="gnn",
                            name="Mateusz Cierpik",
                            log_model=True)
    
    
    K = np.array([[1164.6238115833075, 0.0, 713.5791168212891],
                  [0.0, 1164.6238115833075, 570.9349365234375],
                  [0.0, 0.0, 1.0]])
    
    dataset = LidarGraphDataset("graphs/")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=5, collate_fn=lidar_collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=5, collate_fn=lidar_collate_fn)
    
    model = LidarYOLOXModule(num_classes=8, in_channels=4, K=K)
    trainer = Trainer(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=10, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, loader, val_loader)
    
    wandb.finish()


if __name__ == "__main__":
    main()