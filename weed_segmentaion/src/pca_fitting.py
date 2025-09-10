# This code tries to fit a plane from the output of a stereo node with PCA algorithm

# NOT COMPATIBLE DUE TO A LOT OF NOISE IN THE SCENE - DO NOT USE

# Necessary imports
import numpy as np
import depthai as dai
import open3d as o3d
from sklearn.decomposition import PCA
import time

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

    # Initialize with some dummy point
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

    last_plane_draw_time = time.time()

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
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0,0,1.0], (pts.shape[0], 1)))  # Blue

            # Remove some noise from the point cloud
            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors = 25, std_ratio = 2.0)
            pcd = pcd.select_by_index(ind)

            # Create a PCA classifier and fit the 3D points
            pca = PCA(n_components=3)
            pca.fit(np.asarray(pcd.points))

            # Extract the normal vector and centroid from PCA
            normal_vector = pca.components_[2] 
            centroid = pca.mean_
            
            # Define the plane equation Carthesian parameters
            a,b,c = normal_vector
            d = np.dot(normal_vector, centroid)

            # Draw the detected plane every second
            current_time = time.time()
            if current_time - last_plane_draw_time > 1.0:
                # Create a grid of points centered at the centroid
                plane_size = 1.0  # meters
                grid_n = 20
                xx, yy = np.meshgrid(
                    np.linspace(centroid[0] - plane_size, centroid[0] + plane_size, grid_n),
                    np.linspace(centroid[1] - plane_size, centroid[1] + plane_size, grid_n)
                )
                # Plane equation: ax + by + cz = d  => z = (d - a*x - b*y)/c
                zz = (d - a*xx - b*yy) / c

                # Flatten and stack to get Nx3 points
                plane_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
                pcd_inliers_vis.points = o3d.utility.Vector3dVector(plane_points)
                pcd_inliers_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red

                last_plane_draw_time = current_time

            vis.update_geometry(pcd)
            vis.update_geometry(pcd_inliers_vis)
            vis.poll_events()
            vis.update_renderer()



