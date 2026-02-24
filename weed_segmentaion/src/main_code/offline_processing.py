# This code fetches a single depth frame for processing on command.

# Imports 
import numpy as np
import cv2
import utils
import scipy.ndimage as ndi
import os

# Labels 
FLAT = 0
SLOPE = 1
EDGE = 2

# Thresholds (tune!)
T_VAR_DEPTH = 0.005 # 0 is perfectly flat patch, patches with increased terrain and parts has increased value
T_GRAD_EDGE = 0.006  # 0 is perfectly flat, higher values repersent more tilted patch
theta_flat_thresh = np.radians(60)  # Approximate camera angle

##### OPEN3D INITIALIZATION #####

# Create a O3D visualizer
vis = utils.o3d.visualization.Visualizer()
vis.create_window(window_name="Live preview", width=800, height=600)
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

##############################
# Filepath to depth data
filepath = os.path.join("oak_depth_data", "depth_00000.npy")
#print(os.getcwd())

# Load from saved depth maps
Z = np.load(filepath)

# Conversion from mm to m
Z = Z.astype(np.float32) / 1000.0 
Z = np.ascontiguousarray(Z)
h,w = Z.shape          

# Used to store point cloud data for SLOPE patches only
points_slope = []

# ======== GLOBAL DEPTH MAP PARAMETER ESTIMATION ======== #

# Sobel gradients 3x3
Gx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize = 3)
Gy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize = 3) 

# Gradient magnitude for the entire depth map 1280x720 
grad_mag = np.sqrt(Gx**2 + Gy**2)

# Patch division
patch_size = 40 # in pixels
patches_x = w // patch_size
patches_y = h // patch_size

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


#print(patch_labels) # This look okay

# FORMAT
# [OTHER OTHER FLAT FLAT SLOPE SLOPE
#  OTHER FLAT ..... ]

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

print(valid_slope_mask)

# FORMAT : SLOPE 1 REST 0
# [0 0 1 1 0 0 0 1 1
#  0 0 1 1 1 1 1 0 0....]

for py in range(patches_y):
    for px in range(patches_x):

        # 0 is REST, skip it : 1 is SLOPE, calculate points
        if valid_slope_mask[py, px] == 0:
            continue

        # Looping mechanism
        y0 = py * patch_size
        y1 = y0 + patch_size
        x0 = px * patch_size
        x1 = x0 + patch_size

        # Backprojecting
        Xp, Yp, Zp = utils.backproject_patch(y0,y1,x0,x1,Z)
        pcd_patch = np.stack([Xp.flatten(), Yp.flatten(), Zp.flatten()], axis=1)
        pcd_patch = pcd_patch[np.isfinite(pcd_patch).all(axis=1)]

        points_slope.append(pcd_patch)


# Vertical stacking
points_slope_arr = np.vstack(points_slope)  # Merge all slope patch points

# Filtered data for RANSAC
pcd.points = utils.o3d.utility.Vector3dVector(points_slope_arr)

# RANSAC
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01, # Try adjusting this parameter in case plane starts shifting
    ransac_n=3, 
    num_iterations=1000) # Try calculating it with a given formula

# Plane parameters
a,b,c,d = plane_model

# Update inlier points
pcd_inliers = pcd.select_by_index(inliers)
pcd_inliers.points = pcd_inliers.points
pcd_inliers.colors = pcd_inliers.colors

# Point cloud into array  
# [:,0] => X coordinates // [:,1] => Y coordinates // [:,2] => Z coordinates
inlier_points = np.asarray(pcd_inliers.points)

# Update visualizer and its point clouds
vis.update_geometry(pcd)
vis.update_geometry(pcd_inliers)
vis.poll_events()
vis.update_renderer()
vis.run()