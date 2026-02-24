
# Imports 
import numpy as np
import utils
import os
import matplotlib.pyplot as plt

##### OPEN3D INITIALIZATION #####

# Create a O3D visualizer
vis = utils.o3d.visualization.Visualizer()
vis.create_window(window_name="Live preview", width=800, height=600)
vis.add_geometry(utils.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

# Point cloud variable for SLOPE patch data
pcd = utils.o3d.geometry.PointCloud()
pcd.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red
#vis.add_geometry(pcd)

# Point cloud variable for inliers
# pcd_inliers = utils.o3d.geometry.PointCloud()
# pcd_inliers.points = utils.o3d.utility.Vector3dVector(np.zeros((1,3)))
# pcd_inliers.paint_uniform_color([0.0, 1.0, 0.0]) # Green
# vis.add_geometry(pcd_inliers)

##############################
# Filepath to depth data
filepath = os.path.join("oak_depth_data", "depth4_00151.npy")
#print(os.getcwd())

# Load from saved depth maps
Z = np.load(filepath)

# Conversion from mm to m
Z = Z.astype(np.float32) / 1000.0 
Z = np.clip(Z, 0.0, 3.0)
Z = np.ascontiguousarray(Z)
h,w = Z.shape    

# Colorized depth map
plt.figure(figsize=(8,6))
plt.imshow(Z, cmap='plasma')  # 'plasma', 'viridis', 'gray', etc.
plt.colorbar(label='Depth (m)')
plt.title('Depth Map')
#plt.show()

# print("Z min/max:", Z.min(), Z.max())
# Plot histogram
# plt.figure(figsize=(8,4))
# plt.hist(Z.flatten(), bins=100, color='blue', alpha=0.7)
# plt.title("Distribution of valid depth values (meters)")
# plt.xlabel("Depth (m)")
# plt.ylabel("Number of pixels")
# plt.show()


# Backprojecting step
X,Y = utils.backproject(h,w,Z)

# Convert to point cloud for RANSAC
pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0)
pts = pts[valid]

# Filtered data for RANSAC
pcd.points = utils.o3d.utility.Vector3dVector(pts)

# RANSAC
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.05, # Try adjusting this parameter in case plane starts shifting
    ransac_n=3, 
    num_iterations=1500) # Try calculating it with a given formula

# Plane parameters
a,b,c,d = plane_model

# Update inlier points
pcd_inliers = pcd.select_by_index(inliers)
pcd_inliers.paint_uniform_color([0.0, 1.0, 0.0])  # Green for inliers
vis.add_geometry(pcd_inliers)

# If outliers visual is needed
# pcd_outliers = pcd.select_by_index(inliers, invert=True)
# pcd_outliers.paint_uniform_color([1.0, 0.0, 0.0]) # Red for outliers
# vis.add_geometry(pcd_outliers)

print(f"Plane parameters : a : {a}, b : {b}, c : {c}, d : {d}.")
print(f"Inliers ratio : {(len(inliers) / pts.shape[0])*100} %")

## Boundary line fitting 
inlier_points = np.asarray(pcd_inliers.points)
X, Y, Z = inlier_points[:,0], inlier_points[:,1], inlier_points[:,2]

thickness = 0.02
Y_max = Y.max()
Y_min = Y.min()

# Y coordinates of 'extremes' points
Y_top_points    = Y >= Y_max - thickness
Y_bottom_points = Y <= Y_min + thickness

# Same points as in Y_top_points but with all 3 coordinates
top_points = inlier_points[Y_top_points]
bottom_points = inlier_points[Y_bottom_points]

top_line_coeff = np.polyfit(top_points[:,0], top_points[:,2], 1) # Line is fitted in the x,z plane
bottom_line_coeff = np.polyfit(bottom_points[:,0], bottom_points[:,2],1)

# Create line points for visualization
x_line = np.linspace(X.min(), X.max(), 100)
z_top_line = top_line_coeff[0]*x_line + top_line_coeff[1]
z_bottom_line = bottom_line_coeff[0]*x_line + bottom_line_coeff[1]
y_line = np.full_like(x_line, Y_max)      # top line
y_line_bottom = np.full_like(x_line, Y_min)  # bottom line

# Stack into point clouds
top_line_pts = np.stack([x_line, y_line, z_top_line], axis=1)
bottom_line_pts = np.stack([x_line, y_line_bottom, z_bottom_line], axis=1)

# Convert to Open3D point clouds
top_line_pcd = utils.o3d.geometry.PointCloud()
top_line_pcd.points = utils.o3d.utility.Vector3dVector(top_line_pts)
top_line_pcd.paint_uniform_color([1,1,0])  # Yellow

bottom_line_pcd = utils.o3d.geometry.PointCloud()
bottom_line_pcd.points = utils.o3d.utility.Vector3dVector(bottom_line_pts)
bottom_line_pcd.paint_uniform_color([0,0,1])  # Blue

# Update visualizer and its point clouds
vis.update_geometry(pcd)
vis.update_geometry(pcd_inliers)
#vis.update_geometry(pcd_outliers)
vis.add_geometry(top_line_pcd)
vis.add_geometry(bottom_line_pcd)
vis.poll_events()
vis.update_renderer()
vis.run()