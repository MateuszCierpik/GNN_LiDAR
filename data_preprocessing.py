from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import open3d as op3
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KDTree
from torch_geometric.data import Data
import torch

def project_points(K, X, Y, Z):
    u = K[0,0] * X / Z + K[0,2]
    v = K[1,1] * Y / Z + K[1,2]
    return u, v

class DataPreprocessing:
    def __init__(self):
        self.frame_idx = 0
        self.pointfiled_dtype = np.dtype([
            ('x', np.float32, 1),
            ('y', np.float32, 1),
            ('z', np.float32, 1),
            ('intensity', np.float32, 1),
            ('ring', np.uint16, 1),
            ('time', np.float32, 1),
        ])       
                
        self.T_lidar_camRect1 = np.array([
            [ 0.01539728189227399,  -0.0012823052573279758,   0.9998806325774878,   0.448 ],
            [-0.9996610000153124,    0.020978176075891836,    0.015420803380972237, 0.255 ],
            [-0.02099544614233234,  -0.9997791115150167,     -0.0009588636652390625, -0.215 ],
            [ 0.0,                   0.0,                     0.0,                   1.0 ]
        ])
  
        #   camRect1:
        #     camera_type: frame
        #     camera_location: left
        #     is_rectified: true
        #     camera_model: pinhole
        #     camera_matrix:
        #     - 1164.6238115833075
        #     - 1164.6238115833075
        #     - 713.5791168212891
        #     - 570.9349365234375
        
        
        self.T_cam_lidar = np.linalg.inv(self.T_lidar_camRect1)
        
        self.K = np.array([[1164.6238115833075, 0.0, 713.5791168212891],
                           [0.0, 1164.6238115833075, 570.9349365234375],
                           [0.0, 0.0, 1.0]])
        
        
        self.tracks = np.load("object_detections/interlaken_00_c/left/tracks.npy")
        self.exposures = self.load_image_timestamps("object_detections/interlaken_00_c/left/interlaken_00_c_image_exposure_timestamps_left.txt") 
        
    def load_image_timestamps(self, file_name):
        timestamps = []
        with open(file_name) as f:
            header = f.readline()
            for line in f:
                t_start, t_end = line.strip().split(", ")
                timestamps.append( (int(t_start), int(t_end)) )
              
        return timestamps
        
        
    def pointcloud2_to_xyzit(self, msg):
        N = msg.width          
        step = msg.point_step
        raw = msg.data

        arr = np.frombuffer(raw, dtype=self.pointfiled_dtype, count=N)

        xyzit = np.stack([arr['x'], arr['y'], arr['z'], arr['intensity'], arr['time']], axis=1)
        return xyzit
        
    def prepare_seq(self, seq_path, out_dir):
        frame_id = 0    
        with AnyReader([seq_path]) as reader:
            for conn, timestamp, raw in reader.messages():
                if conn.topic == '/velodyne_points':
                    msg = reader.deserialize(raw, conn.msgtype)
                    # print(msg)
                    # print(len(msg.data))
                    # print(type(msg.data))
                    # print(msg.data.shape)

                    pts = self.pointcloud2_to_xyzit(msg)
                    filename = out_dir / f"frame_{frame_id:06d}.npy"
                    np.save(filename, pts)
                    
                    frame_id += 1
          
        
    def map_points(self, points_dir, visualize=False):
        history_points = []
        num_back = 1
        for frame_id, name in enumerate(os.listdir(points_dir)):   
            pts = np.load(os.path.join(points_dir, name))

            img = cv2.imread(f"E:/GSN/imgs/{frame_id:06d}.png")
            img_distorted = cv2.imread(f"E:/GSN/imgs_/train/interlaken_00_c/images/left/distorted/{frame_id:06d}.png")
            H, W = img.shape[:2]
            
            xyz = pts[:, :3]
            
            if visualize:
                pcd = op3.geometry.PointCloud()
                pcd.points = op3.utility.Vector3dVector(xyz)
                op3.visualization.draw_geometries([pcd])
            
            ones = np.ones((xyz.shape[0], 1))
            pts_h = np.hstack([xyz, ones])
            pts_cam = (self.T_cam_lidar @ pts_h.T).T
            history_points.append(np.hstack([pts_cam, pts[:,3:4]]))
            if len(history_points) > num_back:
                history_points.pop(0)

            pts_cam_merged = np.concatenate(history_points, axis=0)
            
            X, Y, Z = pts_cam_merged[:, 0], pts_cam_merged[:, 1], pts_cam_merged[:, 2]
            intensity = pts_cam_merged[:, 3]
            
            mask = Z > 0
            X, Y, Z = X[mask], Y[mask], Z[mask]
            intensity = intensity[mask] 
            
            xyz_ = np.stack([X, Y, Z], axis=1)
            print(xyz_.shape)
            edges = self.build_radius_graph(xyz_, radius=0.5, max_neighbors=30)
            nodes = self.build_node_features(xyz_, intensity)
            edge_attr = self.build_edge_features(xyz_, edges)

            # print("Edges:", edges.shape)
            # print("Nodes:", nodes.shape)
            # self.visualize_graph(xyz_vis, edges)
            
            u, v = project_points(self.K, X, Y, Z)            
            mask_img = (u >= 0) & (u < W) & (v >=0) & (v < H)
            u = u[mask_img].astype(int)
            v = v[mask_img].astype(int)
            Z_mask = Z[mask_img]    
 
            Z_min = 0.0
            Z_max = 50.0
 
            Z_clip = np.clip(Z_mask, Z_min, Z_max)
            Z_norm = (Z_clip - Z_min) / (Z_max - Z_min)
            Z_norm = (Z_norm * 255).astype(np.uint8)                
            
            
            mask = (self.tracks["t"] >= self.exposures[frame_id][0]) & (self.tracks["t"] <= self.exposures[frame_id][1])    
            frame_tracks = self.tracks[mask]
            
            if visualize:
                
                ## lidar points    
                fig, axs = plt.subplots(2, 2, figsize=(12,12))
                axs[0,0].scatter(X, Y, s=1)
                axs[0,0].set_title("X Y")
                axs[0,0].axis("equal")

                axs[0,1].scatter(X, Z, s=1)
                axs[0,1].set_title("X Z")
                axs[0,1].axis("equal")

                axs[1,0].scatter(u, v, s=1)
                axs[1,0].set_title("u v")

                axs[1,1].imshow(img[:,:,::-1])
                axs[1,1].scatter(u, v, c=Z_mask, cmap='jet', s=2)
                axs[1,1].set_title("cam view")
                
                plt.tight_layout()
                plt.show()
            
                ## bboxy   
                for det in frame_tracks:
                    x = int(det["x"])
                    y = int(det["y"])
                    w = int(det["w"])
                    h = int(det["h"])
                    cls = int(det["class_id"])
                    track = int(det["track_id"])
                    color = (0, 0, 255)

                    cv2.rectangle(img_distorted, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img_distorted, f"{cls}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow("results", img_distorted)
                cv2.waitKey(0)
                plt.show()
                
                
                
            bboxes = np.stack([
                frame_tracks["x"],
                frame_tracks["y"],
                frame_tracks["w"],
                frame_tracks["h"]
            ], axis=1)


            labels = np.array(frame_tracks["class_id"], ndmin=1)
            bboxes = np.array(bboxes, ndmin=2)
            
            ## save
            data = Data(
                x=torch.tensor(nodes, dtype=torch.float32),
                pos=torch.tensor(xyz_, dtype=torch.float32),
                edge_index=torch.tensor(edges.T, dtype=torch.int64),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                bboxes=torch.tensor(bboxes, dtype=torch.int16),
                labels=torch.tensor(labels, dtype=torch.int16),
            )
            
            torch.save(data, f"graphs/frame_{frame_id:06d}.pt")
            
    
    def build_edge_features(self, points_xyz, edges):
        src = edges[:,0]
        dst = edges[:,1]

        delta = points_xyz[dst] - points_xyz[src]
        dist = np.linalg.norm(delta, axis=1, keepdims=True)

        edge_feat = np.concatenate([delta, dist], axis=1)
        return edge_feat.astype(np.float32)
    
    def build_node_features(self, points_xyz, intensity):
        x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # print(intensity.shape)
        features = np.stack([x, y, z, intensity], axis=1)

        return features.astype(np.float32)

    def build_radius_graph(self, points_xyz, radius=0.5, max_neighbors=30):
        tree = KDTree(points_xyz)
        ind = tree.query_radius(points_xyz, r=radius)
        edges = []

        for i, neighbors in enumerate(ind):
            neighbors = neighbors[neighbors != i]

            if len(neighbors) > max_neighbors:
                d = np.linalg.norm(points_xyz[neighbors] - points_xyz[i], axis=1)
                neighbors = neighbors[np.argsort(d)[:max_neighbors]]

            for j in neighbors:
                edges.append((i, j))

        return np.array(edges)   
    
    def visualize_graph(self, points_xyz, edges, max_edges=20000):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        X = points_xyz[:, 0]
        Y = points_xyz[:, 1]
        Z = points_xyz[:, 2]

        ax.scatter(X, Y, Z, s=2, c='blue')

        if len(edges) > max_edges:
            edges_to_draw = edges[np.random.choice(len(edges), max_edges, replace=False)]
        else:
            edges_to_draw = edges

        for s, t in edges_to_draw:
            xs = [points_xyz[s, 0], points_xyz[t, 0]]
            ys = [points_xyz[s, 1], points_xyz[t, 1]]
            zs = [points_xyz[s, 2], points_xyz[t, 2]]
            ax.plot(xs, ys, zs, c='red', linewidth=0.5, alpha=0.5)

        ax.set_title("graph")
        ax.view_init(elev=0, azim=-90)
        plt.show()
                    
                    
        
    def visualization(self, seq_path):
        vis = op3.visualization.Visualizer()
        vis.create_window("points", width=1280, height=720)

        pcd = op3.geometry.PointCloud()
        initialized = False

        with AnyReader([seq_path]) as reader:
            for conn, timestamp, raw in reader.messages():
                if conn.topic != "/velodyne_points":
                    continue

                msg = reader.deserialize(raw, conn.msgtype)
                pts = self.pointcloud2_to_xyzit(msg)

                if pts.shape[0] == 0:
                    continue

                pcd.points = op3.utility.Vector3dVector(pts[:, :3])

                if not initialized:
                    vis.add_geometry(pcd)
                    initialized = True

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

        vis.destroy_window()


#seq_path_1 = Path(r"E:/GSN/data/interlaken_00/lidar_imu.bag")
#seq_path_1 = Path(r"E:/GSN/data/zurich_city_00/lidar_imu.bag")
out_dir = Path(r"E:/GSN/frames/interlaken_00")
out_dir.mkdir(parents=True, exist_ok=True)
  
dp = DataPreprocessing()
#dp.prepare_seq(seq_path_1, out_dir)    
#dp.visualization(seq_path_1)

dp.map_points(Path(r"E:/GSN/frames/interlaken_00"), visualize=False)

