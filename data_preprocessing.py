from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import open3d as op3
import cv2
import matplotlib.pyplot as plt

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
        
        
        self.tracks = np.load("E:/GSN/train/interlaken_00_c/object_detections/left/tracks.npy")
        self.exposures = self.load_image_timestamps("interlaken_00_c_image_exposure_timestamps_left.txt") 
        
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
        self.frame_id = 0
        with AnyReader([seq_path]) as reader:
            for conn, timestamp, raw in reader.messages():
                if conn.topic == '/velodyne_points':
                    msg = reader.deserialize(raw, conn.msgtype)
                    # print(msg)
                    # print(len(msg.data))
                    # print(type(msg.data))
                    # print(msg.data.shape)

                    pts = self.pointcloud2_to_xyzit(msg)
                    filename = out_dir / f"frame_{self.frame_id:06d}.npy"
                    #np.save(filename, pts)
                    
                    # W = 640 #1440
                    # H = 480 #1080
                    #img = np.zeros((H,W,3))
                    #img = cv2.imread(f"E:/GSN/imgs/{self.frame_id:06d}.png")
                    
                    img = cv2.imread(f"E:/GSN/imgs_/train/interlaken_00_c/images/left/distorted/{self.frame_id:06d}.png")
                    H, W = img.shape[:2]
                    
                    xyz = pts[:, :3]
                    pcd = op3.geometry.PointCloud()
                    pcd.points = op3.utility.Vector3dVector(xyz)
                    op3.visualization.draw_geometries([pcd])

                    plt.figure(figsize=(8,8))
                    plt.scatter(pts[:, 0], pts[:, 1], s=0.2)
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title("widok z gory")
                    plt.axis("equal")
                    
                    ones = np.ones((xyz.shape[0], 1))
                    pts_h = np.hstack([xyz, ones])
                    pts_cam = (self.T_cam_lidar @ pts_h.T).T
                    
                    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
                    mask = Z > 0
                    X, Y, Z = X[mask], Y[mask], Z[mask]
                    
                    plt.figure(figsize=(8,8))
                    plt.scatter(X, Z, s=0.2)
                    plt.xlabel("X")
                    plt.ylabel("Z")
                    plt.title("cam view")
                    plt.axis("equal")
                    
                    plt.figure(figsize=(8,8))
                    plt.scatter(X, Y, s=0.2)
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title("cam view 2")
                    plt.axis("equal")
                    
                    
                    u = self.K[0,0] * X / Z + self.K[0,2]
                    v = self.K[1,1] * Y / Z + self.K[1,2]
                    
                    mask_img = (u >= 0) & (u < W) & (v >=0) & (v < H)
                    u = u[mask_img].astype(int)
                    v = v[mask_img].astype(int)
                    for px, py in zip(u, v):
                        cv2.circle(img, (px, py), 2, (0,255,0), -1)
                        
                        
                        
                    ## bboxy 
                    mask = (self.tracks["t"] >= self.exposures[self.frame_id][0]) & (self.tracks["t"] <= self.exposures[self.frame_id][1])    
                    frame_tracks = self.tracks[mask]    
                    for det in frame_tracks:
                        x = int(det["x"])
                        y = int(det["y"])
                        w = int(det["w"])
                        h = int(det["h"])
                        cls = int(det["class_id"])
                        track = int(det["track_id"])
                        color = (0, 0, 255)

                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, f"{cls}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.imshow("results", img)
                    cv2.waitKey(0)
                    plt.show()
                    
                    self.frame_id += 1
                    
                    
                    

        #print(self.frame_id)
        #print(pts[0])
        #print(pts.shape)
        
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


seq_path_1 = Path(r"E:/GSN/data/interlaken_00/lidar_imu.bag")
#seq_path_1 = Path(r"E:/GSN/data/zurich_city_00/lidar_imu.bag")
out_dir = Path(r"frames_00")
out_dir.mkdir(parents=True, exist_ok=True)
  
dp = DataPreprocessing()
dp.prepare_seq(seq_path_1, out_dir)    
#dp.visualization(seq_path_1)

