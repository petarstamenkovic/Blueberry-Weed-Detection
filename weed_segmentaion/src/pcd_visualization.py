# This code snippet allows live visualization of a 3d point cloud data

# Necessary imports
import numpy as np
import depthai as dai
import open3d as o3d

# Function to get stereo pair
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

# Function to close a visualizer
def close_visualizer(vis):
    vis.destroy_window()
    print("Visualizer closed.")

# Create a pipeline
pipeline = dai.Pipeline()

# Setup mono cameras
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth
stereo = getStereoPair(pipeline, monoLeft, monoRight)
xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("Depth")
stereo.depth.link(xOutDepth.input)

# Camera intrinsics
fx, fy = 570.33, 570.3
cx, cy = 655.38, 358.85

def depth_to_points(depth_mm, fx, fy, cx, cy):
    D = depth_mm.astype(np.float32) / 1000.0
    h, w = D.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * D / fx
    Y = (v - cy) * D / fy
    Z = D
    valid = (D > 0) & np.isfinite(D)
    return np.stack([X[valid], Y[valid], Z[valid]], axis=1)

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)

    # Setup Open3D visualizer once
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live 3D", width=800, height=600)

    # Initialize with some dummy point (avoid empty point cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    while True:
        depth = depthQueue.get()
        depthFrame = depth.getFrame()

        pts = depth_to_points(depthFrame, fx, fy, cx, cy)

        if pts.shape[0] < 10:
            print("[INFO] Not enough valid points, skipping this frame...")
            continue 

        limit = 100000
        if pts.shape[0] > limit:
            idx = np.random.choice(pts.shape[0], limit, replace=False)
            pts = pts[idx]

        # Point cloud update
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5,0.5,1.0], (pts.shape[0], 1)))  # optional color

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Terminate the program with Ctrl + C in terminal

