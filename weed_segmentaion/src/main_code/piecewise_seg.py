# This code uses a built LO-RANSAC to enhance the fitted plane.

# Imports 
import numpy as np
import utils

# Build a pipeline
pipeline = utils.dai.Pipeline()
stereo = utils.buildStereoPipeline(pipeline)

# XLinkOut node for depth output
xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("Depth")
stereo.depth.link(xOutDepth.input)

# Connect to device
with utils.dai.Device(pipeline) as device:

    # Fetch an output from XLinkOut node
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)

    # Create a O3D visualizer
    vis = utils.o3d.visualization.Visualizer()
    vis.create_window(window_name="Live preview", width=800, height=600)
    vis.add_geometry(utils.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # All points from point cloud initialization
    pcd = utils.o3d.geometry.PointCloud()
    pcd.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.paint_uniform_color([1.0, 1.0, 0.0]) # Black
    vis.add_geometry(pcd)

    # Plane inlies point cloud initialization
    pcd_inliers_vis = utils.o3d.geometry.PointCloud()
    pcd_inliers_vis.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_inliers_vis.paint_uniform_color([1.0, 0.0, 0.0]) # Red
    vis.add_geometry(pcd_inliers_vis)

    # Lower line of intersection point cloud initialization
    pcd_line_set_lower = utils.o3d.geometry.LineSet()
    pcd_line_set_lower.points = utils.o3d.utility.Vector3dVector(np.zeros((2,3)))
    pcd_line_set_lower.lines = utils.o3d.utility.Vector2iVector([[0,1]])
    pcd_line_set_lower.colors = utils.o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]]) # Green
    vis.add_geometry(pcd_line_set_lower)

    # Upper line of intersection point cloud initialization
    pcd_line_set_upper = utils.o3d.geometry.LineSet()
    pcd_line_set_upper.points = utils.o3d.utility.Vector3dVector(np.zeros((2,3)))
    pcd_line_set_upper.lines = utils.o3d.utility.Vector2iVector([[0,1]])
    pcd_line_set_upper.colors = utils.o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]]) # Blue
    vis.add_geometry(pcd_line_set_upper)

    # Main loop for processing
    while True:
        # Fetch a depth frame
        depth = depthQueue.get()
        depthFrame = depth.getFrame()

        # Conversion from mm to m
        Z = depthFrame / 1000.0
        h,w = Z.shape

        # Resolution is 1280x720
        #print("Resolution : ", Z.shape)

        # This should result in 16 * 9 = 144 patches with the size of (80x80 pixels)
        patch_size = 80
        patches_x = w / patch_size
        patches_y = h / patch_size

        
        
        # 2D depth data -> 3D Point cloud data
        X,Y = utils.backproject(h,w,Z)

        # Convert to point cloud for RANSAC - 2D arrays (X(u,v), Y(u,v) and Z(u,v)) into 1D vector (x0,y0,z0)
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0) # In case of missing depth calculation
        pts = pts[valid]

        # RANSAC needs at least 3 points for plane fitting
        if pts.shape[0] < 3:
            continue

        # Filtered point cloud ready for RANSAC fitting - Black
        pcd.points = utils.o3d.utility.Vector3dVector(pts)

        # Downsample for live preview
        if pts.shape[0] > 500:
            idx = np.random.choice(pts.shape[0], 5000, replace=False)
            pts_lo_ransac = pts[idx]
        else:
            pts_lo_ransac = pts

        # RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02, # Try adjusting this parameter in case plane starts shifting
            ransac_n=3, 
           num_iterations=1000) # Try calculating it with a given formula
 
        # Plane parameters
        a,b,c,d = plane_model

        # Update inlier points
        pcd_inliers = pcd.select_by_index(inliers)
        pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_inliers.points
        pcd_inliers_vis.colors = pcd_inliers.colors

        # Update visualizer and its point clouds
        vis.update_geometry(pcd)
        vis.update_geometry(pcd_inliers_vis)
        #vis.update_geometry(pcd_line_set_lower)
        #vis.update_geometry(pcd_line_set_upper)
        vis.get_render_option().line_width = 10
        vis.poll_events()
        vis.update_renderer()