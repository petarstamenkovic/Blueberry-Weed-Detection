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

# Create pipeline
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

# Connect to device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)

    # Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Slope Inliers & Max Gradient", width=800, height=600)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # Plane inliers (slope)
    pcd_inliers_vis = o3d.geometry.PointCloud()
    pcd_inliers_vis.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_inliers_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    vis.add_geometry(pcd_inliers_vis)

    # Max-gradient point
    max_pcd = o3d.geometry.PointCloud()
    max_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    max_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    vis.add_geometry(max_pcd)

    # Green line parallel to X-axis
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.zeros((2,3)))  # dummy
    line_set.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])  # Green
    vis.add_geometry(line_set)

    while True:
        depth = depthQueue.get()
        depthFrame = depth.getFrame().astype(np.float32)
        Z = depthFrame / 1000.0
        h, w = Z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Convert to point cloud for RANSAC
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0)
        pts = pts[valid]

        if pts.shape[0] < 10:
            continue

        # RANSAC Plane fitting
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model

        # Extract inliers
        pcd_inliers = pcd.select_by_index(inliers)
        pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_inliers.points
        pcd_inliers_vis.colors = pcd_inliers.colors

        # Height map relative to plane (1D)
        X_in = np.asarray(pcd_inliers.points)[:,0]
        Y_in = np.asarray(pcd_inliers.points)[:,1]
        Z_in = np.asarray(pcd_inliers.points)[:,2]
        height_vals = a*X_in + b*Y_in + c*Z_in + d

        # 1D gradient along inliers
        grad_mag_1d = np.abs(np.diff(height_vals, n=1))
        grad_mag_1d = np.pad(grad_mag_1d, (0,1), 'constant')

        # Maximum gradient point
        max_idx = np.argmax(grad_mag_1d)
        max_point_3d = np.array([[X_in[max_idx], Y_in[max_idx], Z_in[max_idx]]])
        max_pcd.points = o3d.utility.Vector3dVector(max_point_3d)

        # Green line along X-axis through max-gradient point
        x_range = np.linspace(-1.0, 1.0, 50)  # adjust to scene size
        y_fixed = max_point_3d[0,1]
        z_fixed = max_point_3d[0,2]
        line_points = np.stack([x_range, np.full_like(x_range, y_fixed), np.full_like(x_range, z_fixed)], axis=1)
        line_set.points = o3d.utility.Vector3dVector(line_points)
        lines = [[i, i+1] for i in range(len(line_points)-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(lines),1)))

        # Update visualization
        vis.update_geometry(pcd_inliers_vis)
        vis.update_geometry(max_pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
