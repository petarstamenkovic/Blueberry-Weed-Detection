# This code uses gradient and point cloud differences to find the line of intersection

# Necessary imports
import numpy as np
import depthai as dai
import open3d as o3d
import cv2

# Parameters 
normal_vector_threshold = 0.1
offset_threshold = 0.1

# Camera intrinsics - Extracted from calibration.py
fx, fy = 570.33, 570.3   # Focal lenghts
cx, cy = 655.38, 358.85  # Camera principle values 

# ========================= FUNCTIONS ========================= #

# Function to get stereo pair
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

def updatePlane(prevPlane, newPlane):
    # Current plane Carthesian parameters and normal vector
    a,b,c,d = newPlane
    new_normal_vector = np.array([a,b,c])
    new_normal_vector = new_normal_vector / np.linalg.norm(new_normal_vector)

    # In first iteration, there is no 'previous' plane
    if prevPlane is None:
        return newPlane
    
    # Previous plane Carthesian parameters and normal vector
    prev_a, prev_b, prev_c, prev_d = prevPlane
    old_normal_vector = np.array([prev_a, prev_b, prev_c])
    old_normal_vector = old_normal_vector / np.linalg.norm(old_normal_vector)

    # Calculate the similarities of the normal vectors of a previous and a current plane
    normal_vectors_alligned = abs(np.dot(new_normal_vector, old_normal_vector))

    # Calculate the difference in offsets of planes
    offset_diffence = abs(d - prev_d)

    # Comparison of thep lanes under conditions
    if normal_vectors_alligned > (1 - normal_vector_threshold) and offset_diffence < offset_threshold:
        return newPlane  # Update the plane with a new one 
    else:
        return prevPlane # Keep the previous plane

# Fetch maximum gradient points
def fetch_maximum_points(X, Y, Z, gradient_array, num_points):
    # Get indices of the top 'num_points' gradients
    top_indices = np.argsort(gradient_array)[::-1]
    top_indices = top_indices[:num_points]

    # Fetch corresponding 3D points
    max_points = np.stack([X[top_indices], Y[top_indices], Z[top_indices]], axis=1)

    return max_points
    
# PCA line fitting function
def pca_line_fitting(points):

    # Center the points around the mean
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point

    # Define a covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    eigvals, eigvecs = np.linalg.eig(covariance_matrix)

    direction = eigvecs[:, np.argmax(eigvals)]

    return mean_point, direction

# Convert PCA lines to Open3D lines
def line_to_o3d_line(line_set, mean_point, direction, length=0.5, color=[0,1,0]):
    direction = direction / np.linalg.norm(direction)

    # Create endpoints of the line segment
    endpoint1 = mean_point - (direction * length / 2)
    endpoint2 = mean_point + (direction * length / 2)

    # Define line parameters
    line_points = [endpoint1, endpoint2] # Actual points of the line
    lines = [[0, 1]]  # Single line connecting the two points - index based

    # Add line parameters
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set

# ============================= CODE EXECUTION ============================= #

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

# XLinkOut node for depth output
xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("Depth")
stereo.depth.link(xOutDepth.input)

# Initial plane variable
previousPlane = None

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

    # Lower line of intersection
    line_set_lower = o3d.geometry.LineSet()
    line_set_lower.points = o3d.utility.Vector3dVector(np.zeros((2,3)))
    line_set_lower.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set_lower.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
    vis.add_geometry(line_set_lower)

    # Upper line of intersection
    line_set_upper = o3d.geometry.LineSet()
    line_set_upper.points = o3d.utility.Vector3dVector(np.zeros((2,3))) 
    line_set_upper.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set_upper.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue
    vis.add_geometry(line_set_upper)
 
    # Main loop for processing
    while True:
        # Fetch a depth frame (metrics in mm) and convert it for precission
        depth = depthQueue.get()
        depthFrame = depth.getFrame()#.astype(np.float32) 
        
        # From mm to m
        Z = depthFrame / 1000.0 
        
        ## Noise removal with median filtering
        depthFrame = cv2.medianBlur(depthFrame, 5)

        # From pixel values to real-life values - Back-projecting
        h, w = Z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # So now 
        # X - Real-life x coordinates of a pixel(u,v)
        # Y - Real-life y coordinates of a pixel(u,v)
        # Z - Distance from a camera to a  pixel(u,v)

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

        # Reducing a noise - Both ways are extremely heavy on visualizer - Inresponsive
        #pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors = 20, std_ratio = 2.0)
        #pcd = pcd.select_by_index(ind)

        #cl, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.05)
        #pcd_clean = pcd.select_by_index(ind)

        # Try to segment a plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, # Try adjusting this parameter in case plane starts shifting
            ransac_n=3, 
            num_iterations=1000)

        # Function that keeps a slope plane locked
        activePlane = updatePlane(previousPlane, plane_model)

        # Extract Cartesian parameter values
        a, b, c, d = activePlane
        previousPlane = activePlane

        # Extract inliers and prepare them for visualization 
        pcd_inliers = pcd.select_by_index(inliers)

        # DBScan clustering - Extremely heavy on the visualizer that it becomes inresponsive
        #labels = np.array(
        #    pcd_inliers.cluster_dbscan(eps = 0.05, min_points = 10)
        #)
        #max_label = labels.max()
    
        pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_inliers.points
        pcd_inliers_vis.colors = pcd_inliers.colors

        # X_in - Real-life x coordinates of inlier points
        # Y_in - Real-life y coordinates of inlier points
        # Z_in - Distance from the camera of inlier points
        X_in = np.asarray(pcd_inliers.points)[:,0]
        Y_in = np.asarray(pcd_inliers.points)[:,1]
        Z_in = np.asarray(pcd_inliers.points)[:,2]

        height_values = a*X_in + b*Y_in + c*Z_in + d

        # ============= DIVISION OF A FRAME INTO LOWER AND UPPER ============== #

        # Divison by Y coordinates 
        y_threshold = np.median(Y_in)

        # Create masks for upper and lower inlier points
        lower_frame_mask = Y_in < y_threshold
        upper_frame_mask = Y_in > y_threshold

        # Apply masks
        X_lower, Y_lower, Z_lower = X_in[lower_frame_mask], Y_in[lower_frame_mask], Z_in[lower_frame_mask]
        X_upper, Y_upper, Z_upper = X_in[upper_frame_mask], Y_in[upper_frame_mask], Z_in[upper_frame_mask]

        # Height functions 
        height_lower = height_values[lower_frame_mask]
        height_upper = height_values[upper_frame_mask]

        # ========= SORT SPATIALLY AND COMPUTE GRADIENTS ========= #
        # Lower section
        sort_lower_indices = np.argsort(Y_lower)
        X_lower = X_lower[sort_lower_indices]
        Y_lower = Y_lower[sort_lower_indices]
        Z_lower = Z_lower[sort_lower_indices]
        height_lower = height_lower[sort_lower_indices]

        # Upper section
        sort_upper_indices = np.argsort(Y_upper)
        X_upper = X_upper[sort_upper_indices]
        Y_upper = Y_upper[sort_upper_indices]
        Z_upper = Z_upper[sort_upper_indices]
        height_upper = height_upper[sort_upper_indices]

        # Compute gradients for both sections
        gradient_lower = np.abs(np.diff(height_lower, n=1))
        gradient_lower = np.pad(gradient_lower, (0,1), 'constant')

        gradient_upper = np.abs(np.diff(height_upper, n=1))
        gradient_upper = np.pad(gradient_upper, (0,1), 'constant')

        # Number of points to fit a line of intersection - keep this number higher than 100
        num_points = 200

        # Fetch maximum gradient points
        max_gradient_lower_points = fetch_maximum_points(X_lower, Y_lower, Z_lower, gradient_lower, num_points)
        max_gradient_upper_points = fetch_maximum_points(X_upper, Y_upper, Z_upper, gradient_upper, num_points)

        # Fit lines using PCA
        mean_lower, direction_lower = pca_line_fitting(max_gradient_lower_points)
        mean_upper, direction_upper = pca_line_fitting(max_gradient_upper_points)

        # Convert lines into open3D lines 
        line_set_lower = line_to_o3d_line(line_set_lower, mean_lower, direction_lower, length=0.5, color=[0,1,0])
        line_set_upper = line_to_o3d_line(line_set_upper, mean_upper, direction_upper, length=0.5, color=[0,0,1])

        # Update visualizer and its point clouds
        vis.update_geometry(pcd_inliers_vis)
        vis.update_geometry(line_set_lower)
        vis.update_geometry(line_set_upper)
        vis.poll_events()
        vis.update_renderer()


# Terminate the code with Ctrl + C