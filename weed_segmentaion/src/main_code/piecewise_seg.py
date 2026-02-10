# This code uses a built LO-RANSAC to enhance the fitted plane.

# Imports 
import numpy as np
import cv2
import utils
import scipy.ndimage as ndi

# Labels 
FLAT = 0
SLOPE = 1
EDGE = 2

# Thresholds (tune!)
T_VAR_DEPTH = 0.005 # 0 is perfectly flat patch, patches with increased terrain and parts has increased value
T_GRAD_EDGE = 0.006  # 0 is perfectly flat, higher values repersent more tilted patch
theta_flat_thresh = np.radians(130)  # Approximate camera angle

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
    #vis.create_window(window_name="Live preview", width=800, height=600)
    vis.add_geometry(utils.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # Point cloud variable for SLOPE patch data
    pcd = utils.o3d.geometry.PointCloud()
    pcd.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.paint_uniform_color([1.0, 1.0, 0.0]) # Yellow
    vis.add_geometry(pcd)

    # Point cloud variable for inliers
    pcd_inliers = utils.o3d.geometry.PointCloud()
    pcd_inliers.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0]) # Blue
    vis.add_geometry(pcd_inliers)

    # Point cloud variable for upper line of intersection
    pcd_upper_line = utils.o3d.geometry.PointCloud()
    pcd_upper_line.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_upper_line.paint_uniform_color([0.0, 1.0, 0.0]) # Green
    vis.add_geometry(pcd_upper_line)

    # Point cloud variable for lower line of intersection
    pcd_lower_line = utils.o3d.geometry.PointCloud()
    pcd_lower_line.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd_lower_line.paint_uniform_color([0.0, 0.0, 1.0]) # Red
    vis.add_geometry(pcd_lower_line)


    # Main loop for processing
    while True:
        # Fetch a depth frame - Resolution 1280x720
        depth = depthQueue.get()
        depthFrame = depth.getFrame()

        # Conversion from mm to m
        Z = depthFrame.astype(np.float32) / 1000.0 # Sobel needs float32
        Z = np.ascontiguousarray(Z)
        h,w = Z.shape

        # Patch division
        patch_size = 40 # in pixels
        patches_x = w // patch_size
        patches_y = h // patch_size

        # Used to store point cloud data for SLOPE patches only
        points_slope = []

        # ======== GLOBAL DEPTH MAP PARAMETER ESTIMATION ======== #

        # Sobel gradients 3x3
        Gx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize = 3)
        Gy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize = 3) 
        
        # Gradient magnitude 1280x720 
        grad_mag = np.sqrt(Gx**2 + Gy**2)

        # ======== PATCH PROCESSING ======== #
        patch_labels = np.zeros((patches_y, patches_x), dtype=np.uint8)
        
        for py in range(patches_y):
            for px in range(patches_x):

                # Looping mechanism
                y0 = py * patch_size
                y1 = y0 + patch_size
                x0 = px * patch_size
                x1 = x0 + patch_size

                # Patch data processing
                patch_depth_map =   Z[y0:y1, x0:x1]
                patch_grad = grad_mag[y0:y1, x0:x1]

                # Valid data in depth map
                valid = patch_depth_map > 0
                num_valid = np.count_nonzero(valid)

                if num_valid < 0.2*patch_depth_map.size:
                    patch_labels[py,px] = FLAT
                    continue

                # Use only valid values
                valid_depth_vals = patch_depth_map[valid]
                valid_grad_vals  = patch_grad[valid]

                # FEATURES
                mean_grad = np.mean(valid_grad_vals)
                variance_grad = np.var(valid_grad_vals) # Variance of elements in NumPy array 
                variance_depth = np.var(valid_depth_vals)

                X_patch, Y_patch, Z_patch = utils.backproject_patch(y0,y1,x0,x1,Z)
                pcd_patch = np.stack([X_patch.flatten(), Y_patch.flatten(), Z_patch.flatten()], axis=1)
                pcd_patch = pcd_patch[np.isfinite(pcd_patch).all(axis=1)]
            
                if pcd_patch.shape[0] >= 3:
                    # PCA to get normal
                    cov = np.cov(pcd_patch.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    normal_patch = eigvecs[:, 0]  # Smallest eigenvector
                    if normal_patch[2] > 0: 
                        normal_patch = -normal_patch
                    # Angle with camera Z-axis
                    theta = np.arccos(np.clip(normal_patch[2], -1.0, 1.0))
                else:
                    theta = 0  # Not enough points, treat as flat

                # Classification
                if mean_grad >= T_GRAD_EDGE and variance_depth >= T_VAR_DEPTH:
                    label = EDGE   # aggressive edges
                else:
                    # Use theta to differentiate flat vs slope
                    if theta < theta_flat_thresh:
                        label = FLAT
                    else:
                        label = SLOPE
                        
                patch_labels[py, px] = label

        # Binary mask [SLOPE = 1, REST = 0]
        slope_mask = (patch_labels == SLOPE).astype(np.uint8)                    

        # Connectivity structure
        s = [[1,1,1],
             [1,1,1],
             [1,1,1]
        ]

        # Labeling NDI function definition
        labeled_array, num_features = ndi.label(slope_mask, structure=s)
        sizes = ndi.sum(slope_mask, labeled_array, range(1,num_features+1))

        # Find valid clusters
        cluster_size = 100
        valid_clusters = [i+1 for i,s in enumerate(sizes) if s>=cluster_size]

        valid_slope_mask = np.isin(labeled_array, valid_clusters)

        # #Visualization
        viz = np.zeros((patches_y, patches_x, 3), dtype=np.uint8)

        # #BGR Format 
        viz[valid_slope_mask] = (0,255,0) # Green
        viz[~valid_slope_mask] = (0,0,0) # Black

        viz = cv2.resize(viz, (w,h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Filtered only SLOPE data", viz)
        cv2.waitKey(1)

        points_slope = []

        for py in range(patches_y):
            for px in range(patches_x):

                if valid_slope_mask[py, px] == 0:
                    continue

                y0 = py * patch_size
                y1 = y0 + patch_size
                x0 = px * patch_size
                x1 = x0 + patch_size

                Xp, Yp, Zp = utils.backproject_patch(y0,y1,x0,x1,Z)
                pcd_patch = np.stack([Xp.flatten(), Yp.flatten(), Zp.flatten()], axis=1)
                pcd_patch = pcd_patch[np.isfinite(pcd_patch).all(axis=1)]

                points_slope.append(pcd_patch)
        

        # #Visualization
        #viz = np.zeros((patches_y, patches_x, 3), dtype=np.uint8)

        # #BGR Format 
        # viz[patch_labels == FLAT] = (0,255,255) # Yellow
        # viz[patch_labels == SLOPE] = (0,255,0) # Green
        # viz[patch_labels == EDGE] = (255,0,0) # Blue

        #viz = cv2.resize(viz, (w,h), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("Patch segmentation", viz)
        #cv2.waitKey(1)

        if len(points_slope) == 0:
            #print("[INFO] No SLOPE data!")
            continue

        # Vertical stacking
        points_slope_arr = np.vstack(points_slope)  # Merge all slope patch points

        # RANSAC needs at least 3 points for plane fitting
        if points_slope_arr.shape[0] < 3:
            continue

        # Filtered data for RANSAC
        pcd.points = utils.o3d.utility.Vector3dVector(points_slope_arr)

        # RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02, # Try adjusting this parameter in case plane starts shifting
            ransac_n=3, 
            num_iterations=1000) # Try calculating it with a given formula
 
        # Plane parameters
        a,b,c,d = plane_model

        # Update inlier points
        pcd_inliers = pcd.select_by_index(inliers)
        #pcd_inliers.paint_uniform_color([1.0, 1.0, 0.0])
        pcd_inliers.points = pcd_inliers.points
        pcd_inliers.colors = pcd_inliers.colors

        # Point cloud into array  
        # [:,0] => X coordinates // [:,1] => Y coordinates // [:,2] => Z coordinates
        inlier_points = np.asarray(pcd_inliers.points)

        # Binning method - divide X axis into columns (bins)
        num_bins = 50
        x_min, x_max = inlier_points[:,0].min(), inlier_points[:,0].max()
        bins = np.linspace(x_min, x_max, num_bins+1)

        # Prepare arrays to be fitted
        upper_bin_values = []
        lower_bin_values = []
        x_bin_values     = []

        for i in range(num_bins):
            # Bin edges
            x0, x1 = bins[i], bins[i+1]
            bin_mask = (inlier_points[:,0] >= x0) & (inlier_points[:,0] < x1)
            bin_points = inlier_points[bin_mask]

            if len(bin_points) == 0:
                continue 

            # Sort the Y values
            sorted_y_values = np.sort(bin_points[:,1]) # ascending order

            # Fetch points 
            num_point_to_evaluate = 20
            lower_y_points = sorted_y_values[:num_point_to_evaluate]
            upper_y_points = sorted_y_values[-num_point_to_evaluate:]

            # Median calculation
            lower_median = np.median(lower_y_points)
            upper_median = np.median(upper_y_points)

            # Add to data pool for fitting
            lower_bin_values.append(lower_median)
            upper_bin_values.append(upper_median)

            x_center = (x0+x1)/2
            x_bin_values.append(x_center)

        # Polyfit needs arrays instead of lists
        x_vals = np.array(x_bin_values)
        y_upper = np.array(upper_bin_values)
        y_lower = np.array(lower_bin_values)

        # Linear regression to fit lines - returns slope and intercept
        lower_line_coef = np.polyfit(x_vals, y_lower, 1)
        upper_line_coef = np.polyfit(x_vals, y_upper, 1)

        num_line_points = 50
        x_line = np.linspace(x_vals.min(), x_vals.max(), num_line_points)

        # Compute corresponding y values
        y_lower_line = lower_line_coef[0]*x_line + lower_line_coef[1]
        y_upper_line = upper_line_coef[0]*x_line + upper_line_coef[1]

        # Compute Z on plane
        z_lower_line = (-a*x_line - b*y_lower_line - d)/c
        z_upper_line = (-a*x_line - b*y_upper_line - d)/c

        # Update lines point clouds
        pcd_lower_line.points = utils.o3d.utility.Vector3dVector(np.stack([x_line, y_lower_line, z_lower_line], axis=1))
        pcd_upper_line.points = utils.o3d.utility.Vector3dVector(np.stack([x_line, y_upper_line, z_upper_line], axis=1))

        # Update visualizer and its point clouds
        vis.update_geometry(pcd)
        vis.update_geometry(pcd_inliers)
        vis.update_geometry(pcd_lower_line)
        vis.update_geometry(pcd_upper_line)
        vis.poll_events()
        vis.update_renderer()