# This code tries to fit a plane from the output of a stereo node with PCA algorithm

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

# Function that converts depth map into a 3D point cloud
def depth_to_points(depth_mm, fx, fy, cx, cy):
    D = depth_mm.astype(np.float32) / 1000.0
    h, w = D.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * D / fx
    Y = (v - cy) * D / fy
    Z = D
    valid = (D > 0) & np.isfinite(D)
    return np.stack([X[valid], Y[valid], Z[valid]], axis=1)

# Connecting a device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)

    # Setup Open3D visualizer once
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live 3D", width=800, height=600)

    # Initialize with some dummy point (avoid empty point cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # Create a point cloud for inliers
    pcd_inliers_vis = o3d.geometry.PointCloud()
    pcd_inliers_vis.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    vis.add_geometry(pcd_inliers_vis)

    # Create a point cloud for flat inliers
    pcd_inliers_flat_vis = o3d.geometry.PointCloud()
    pcd_inliers_flat_vis.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    vis.add_geometry(pcd_inliers_flat_vis)

    # Main loop
    while True:
        depth = depthQueue.get()
        depthFrame = depth.getFrame()

        pts = depth_to_points(depthFrame, fx, fy, cx, cy)

        if pts.shape[0] < 10:
            print("[INFO] Not enough valid points, skipping this frame...")
            continue 
        else:

            limit = 50000
            if pts.shape[0] > limit:
                idx = np.random.choice(pts.shape[0], limit, replace=False)
                pts = pts[idx]

            # Point cloud update
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5,0.5,1.0], (pts.shape[0], 1)))  # optional color

            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors = 25, std_ratio = 2.0)
            pcd = pcd.select_by_index(ind)

            # RANSAC Plane fitting - Returns plane parameters from Carthesian plane equation
            # ax + by + cz + d = 0
            plane_model, inliers = pcd.segment_plane(
                distance_threshold = 0.01, # 1cm - Adjust this (increase) if fitting fails
                ransac_n = 3,
                num_iterations = 1000
            )

            # Extract parameters for Carthesian plane equation - slope plane
            a,b,c,d = plane_model
            n = np.array([a,b,c], dtype=np.float64)
            n = n / np.linalg.norm(n)

            # Extract inliers
            pcd_inliers = pcd.select_by_index(inliers)
            pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0]) # Red

            pcd_inliers_vis.points = pcd_inliers.points
            pcd_inliers_vis.colors = pcd_inliers.colors

            # Extract outliers 
            pcd_outliers = pcd.select_by_index(inliers, invert = True)
            pcd_outliers.paint_uniform_color([0.0, 1.0, 0.0]) # Green

            # Segment the flat plane - Find new inliers from the outliers of the previous segmentation
            plane_model_flat, inliers_flat = pcd_outliers.segment_plane(
                distance_threshold = 0.01, 
                ransac_n = 3,
                num_iterations = 1000
            )

            # Extract new flat inliers FROM outliers of the previous segmentation
            pcd_inliers_flat = pcd_outliers.select_by_index(inliers_flat)
            pcd_inliers_flat.paint_uniform_color([0.0, 0.0, 1.0])

            pcd_inliers_flat_vis.points = pcd_inliers_flat.points
            pcd_inliers_flat_vis.colors = pcd_inliers_flat.colors

            # Extract parameters for Carthesian plane equation - flat plane
            a2,b2,c2,d2 = plane_model_flat
            n2 = np.array([a2,b2,c2], dtype=np.float64)
            n2 = n2 / np.linalg.norm(n)

            print(f"Inliers plane parameters  -> a :  {a}, b :   {b}, c :   {c}, d :   {d}")
            print(f"Outliers plane parameters -> a2 : {a2}, b2 : {b2}, c2 : {c2}, d2 : {d2}")
            print("==========================================")

            vis.update_geometry(pcd_inliers_vis)
            vis.update_geometry(pcd_inliers_flat_vis)
            vis.poll_events()
            vis.update_renderer()

        # Terminate the program with Ctrl + C in terminal

