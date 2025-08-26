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

    # Create a Open3D visuzalizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live stream", width=800, height=600)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # Plane inliers (slope)
    pcd_inliers_vis = o3d.geometry.PointCloud()
    pcd_inliers_vis.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_inliers_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    vis.add_geometry(pcd_inliers_vis)

    # Maximum gradient point - intersection of slope and flat plane
    max_pcd = o3d.geometry.PointCloud()
    max_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    max_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    vis.add_geometry(max_pcd)

    # Green line (LineSet) parallel to X-axis 
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.zeros((2,3)))  # 2,3 - Two points in coordinate center initially 
    line_set.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])  # Green
    vis.add_geometry(line_set)

    while True:
        # Fetch a depth frame (metrics in mm) and convert it for precission
        depth = depthQueue.get()
        depthFrame = depth.getFrame()#.astype(np.float32) 
        
        # From mm to m
        Z = depthFrame / 1000.0 
        
        # From pixel values to real-life values - Back-projecting
        h, w = Z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # So now 
        # X - Real-life x coordiantes of a pixel(u,v)
        # Y - Real-life y coordinates of a pixel(u,v)
        # Z - Distance from a cmaera to a  pixel(u,v)

        # Convert to point cloud for RANSAC
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0)
        pts = pts[valid]

        # RANSAC must have enough (at least 3 points) to run
        if pts.shape[0] < 10:
            continue

        # RANSAC Plane fitting
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Try to segment a plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, 
            ransac_n=3, 
            num_iterations=1000)
        
        # Extract Carthesian parameter values
        a, b, c, d = plane_model

        # Extract inliers and prepare them for visualization 
        pcd_inliers = pcd.select_by_index(inliers)
        pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_inliers.points
        pcd_inliers_vis.colors = pcd_inliers.colors

        # X_in - Real-life x coordinates of inlier points
        # Y_in - Real-life y coordinates of inlier points
        # Z_in - Distance from the camera of inlier points
        X_in = np.asarray(pcd_inliers.points)[:,0]
        Y_in = np.asarray(pcd_inliers.points)[:,1]
        Z_in = np.asarray(pcd_inliers.points)[:,2]
        
        # Height map - Depth map does not work since its too much camera depended
        # Relative to a slope normal - distance to a slope plane due to imperfection of real-life slopes
        # Points that are almost beyond slope is exactly what we are looking for - the transition from slope to flat
        height_values = a*X_in + b*Y_in + c*Z_in + d

        # Height value array allows a simple neighbour difference calculation
        gradient_mag = np.abs(np.diff(height_values, n=1))
        gradient_mag = np.pad(gradient_mag, (0,1), 'constant') # same lenght as height_values

        # Maximum gradient point - biggest height difference
        max_diff = np.argmax(gradient_mag)
        max_point_coord = np.array([[X_in[max_diff], Y_in[max_diff], Z_in[max_diff]]])
        max_pcd.points = o3d.utility.Vector3dVector(max_point_coord)

        # Green line along X-axis through max-gradient point
        x_range = np.linspace(-1.0, 1.0, 50)  # Draw a 50 point line from -1 to 1 (ground plane parallel) 
        
        # Fixed values for maximum gradient point 
        y_fixed = max_point_coord[0,1]
        z_fixed = max_point_coord[0,2]
        
        # Combination of 3 coordinates to create a 3D space for the line
        # full_like fills y_fixed and z_fixed to be the same dimension as x_range
        line_points = np.stack([x_range, np.full_like(x_range, y_fixed), np.full_like(x_range, z_fixed)], axis=1)
        
        # Line parameters and configuration
        line_set.points = o3d.utility.Vector3dVector(line_points)
        lines = [[i, i+1] for i in range(len(line_points)-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(lines),1)))

        # Update visualizer and its point clouds
        vis.update_geometry(pcd_inliers_vis)
        vis.update_geometry(max_pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
