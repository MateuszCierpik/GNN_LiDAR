from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import open3d as op3

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
                    np.save(filename, pts)
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


seq_path_1 = Path(r"E:/GSN/data/zurich_city_00/lidar_imu.bag")
out_dir = Path(r"frames_00")
out_dir.mkdir(parents=True, exist_ok=True)
  
dp = DataPreprocessing()
dp.prepare_seq(seq_path_1, out_dir)    
dp.visualization(seq_path_1)

