# This code calculates slope parameters without heavy visualization. In real-world example there will be a signal that 
# starts the operation, but here it will be utilized with a key press or a timer.

# Necessary imports
import json
import math
import numpy as np
import depthai as dai
import open3d as o3d

# Camera intrinsics - Extracted from calibration.py
fx, fy = 570.33, 570.3   # Focal lenghts
cx, cy = 655.38, 358.85  # Camera principle values 

# ========================= FUNCTIONS ========================= #

# Function to get stereo pair and configure it 
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

# Fetch first num_points number of maximum gradient points
def fetch_maximum_points(X, Y, Z, gradient_array, num_points):
    # Sort in descending order and grab first num_points samples
    top_indices = np.argsort(gradient_array)[::-1]
    top_indices = top_indices[:num_points]

    # Fetch corresponding 3D parameters for these points
    max_points = np.stack([X[top_indices], Y[top_indices], Z[top_indices]], axis=1)
    
    # ============== REMOVING OUTSIDERS FROM MAXIUMUM POINTS FOR MORE ACCURATE PCA FITTING ============== #
    y_values = max_points[:, 1]
    median = np.median(y_values)

    # Keep only close points - set threshold here (currently 0.05)
    mask = np.abs(y_values - median) < 0.05 
    filtered_points = max_points[mask]

    # Security in case of low number of points
    if filtered_points.shape[0] > 10:
        return filtered_points
    else:
        return max_points
    # =================================================================================================== #

# PCA line fitting function
def pca_line_fitting(points):

    # Center the points around the mean
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point

    # Define a covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Eigen vectors 
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)

    direction = eigvecs[:, np.argmax(eigvals)]

    return mean_point, direction

# Function that converts REAL-WORLD 3D point cloud back to 2D and returns PIXEL coordinates in a tuple
def backtrack_to_2d(endpoint):
    X, Y, Z = endpoint
    
    # Skip if Z is 0 due to division
    if Z == 0:
        return None
    
    # Backtrack - Both u and v are in pixel values - these will limit the ROI 
    u = int((X * fx) / Z + cx)
    v = int((Y * fy) / Z + cy)

    return (u,v)

# Convert PCA lines to Open3D lines
def line_to_o3d_line(line_set, mean_point, direction, length=0.5, color=[0,1,0]):
    direction = direction / np.linalg.norm(direction)

    # Create endpoints of the line segment
    endpoint1 = mean_point - (direction * length / 2)
    endpoint2 = mean_point + (direction * length / 2)

    # Define line parameters
    line_points = [endpoint1, endpoint2] # Actual points of the line
    lines = [[0, 1]]  # Single line connecting the two points - index based ; "connect 0-th and 1-st point"

    # Points in 2D - For line equation and ROI configuration later in the process
    point1 = backtrack_to_2d(endpoint1)
    point2 = backtrack_to_2d(endpoint2)

    # Add line parameters
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set, point1, point2

# Function that calculates 2D line equation
def calculate_line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    A = y1 - y2
    B = x2 - x1

    m = round(A / B , 2)

    return m, x1, y1


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

# Connect to device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)



    # Main loop for processing
    while True:

        input("[ACTION REQUIRED] Press Enter to continue...")
        print("[INFO] Fetching slope parameters....")

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

        # Fetch a depth frame (metrics in mm) and convert it for precission
        depth = depthQueue.get()
        depthFrame = depth.getFrame()
        
        # From mm to m
        Z = depthFrame / 1000.0 

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

        #"""
        # Downsample point cloud if too large - necessary for noise reduction 
        if pts.shape[0] > 20000:
            idx = np.random.choice(pts.shape[0], 20000, replace=False)
            pts = pts[idx]
        #"""

        # RANSAC must have enough (at least 3 points) to run
        if pts.shape[0] < 10:
            continue

        # RANSAC Plane fitting
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Reducing a noise 
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors = 20, std_ratio = 2.0)
        pcd = pcd_clean

        pcd_radius, ind_radius = pcd.remove_radius_outlier(nb_points=20, radius=0.1)
        pcd = pcd_radius

        print("[INFO] Segmenting plane...")
        # Try to segment a plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02, # Try adjusting this parameter in case plane starts shifting
            ransac_n=3, 
            num_iterations=1000)

        # Extract Cartesian parameter values
        a, b, c, d = plane_model

        # Extract inliers and prepare them for visualization 
        pcd_inliers = pcd.select_by_index(inliers)

        # ================= DBSCAN Clustering attempt ================= #
        
        print("[INFO] Applying DBSCAN clustering to remove noise...")

        # Apply dbScan to current pcd to remove noise
        labels = np.array(pcd_inliers.cluster_dbscan(eps = 0.02,
                                                     min_points = 35,
                                                     print_progress = True))
        
        max_labels = labels.max()
        print(f"[INFO] Found  {max_labels + 1} clusters in the plane inliers point cloud.")

        # Count how many times each value from 0 to max_labels apperead in labels array (not considering noise -1)
        counts = np.bincount(labels[labels >= 0])

        # Find the index of the biggest cluster
        max_cluster_ind = np.argmax(counts)

        # Fetch the points from the largest cluster
        pcd_clustered = pcd_inliers.select_by_index(np.where(labels == max_cluster_ind)[0])

        # ================ ANGLE OF ROTATION CALCULATION ================ #

        # n = [nx, ny, nz]
        n = np.array([a,b,c], dtype=np.float64)
        n = n / np.linalg.norm(n)

        # Normal point UP since OAK y axis is DOWN
        if n[1] > 0:
            n = -n

        # How much is bank tilted up and down reffereing to x axis?
        # CONSIDERATION - Once the camera angle is fixed you can subtract it from theta_x for the actual tilt of the slope
        theta_x = math.atan2(n[2], -n[1])

        # How much is bank rotated left and right compared to the z axis?
        theta_y = math.atan2(n[0], -n[1])

        # Configure inliers visualization
        pcd_clustered.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_clustered.points
        pcd_inliers_vis.colors = pcd_clustered.colors
        # =============================================================== #

        # X_in - Real-life x coordinates of inlier points
        # Y_in - Real-life y coordinates of inlier points
        # Z_in - Distance from the camera of inlier points
        X_in = np.asarray(pcd_clustered.points)[:,0]
        Y_in = np.asarray(pcd_clustered.points)[:,1]
        Z_in = np.asarray(pcd_clustered.points)[:,2]

        # How much is the point far away from the ideal segmented plane 
        height_values = a*X_in + b*Y_in + c*Z_in + d

        # ============= DIVISION OF A FRAME INTO LOWER AND UPPER ============== #

        # Divison by Y coordinates 
        Y_min, Y_max = np.min(Y_in), np.max(Y_in)

        # Geometrical cut of Y axis
        y_threshold = (Y_min + Y_max) / 2

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

        # ================ MAXIMUM COMPUTING - WITHOUT GRADIENTS ================ #
        """
        gradient_lower = np.argsort(np.abs(height_lower))[::-1][:400]
        max_gradient_lower_points = np.stack([X_lower[gradient_lower], Y_lower[gradient_lower], Z_lower[gradient_lower]], axis=1)  

        gradient_upper = np.argsort(np.abs(height_upper))[::-1][:400]
        max_gradient_upper_points = np.stack([X_upper[gradient_upper], Y_upper[gradient_upper], Z_upper[gradient_upper]], axis=1)
        """
        # ======================================================================= # 

        # Compute gradients for both sections
        gradient_lower = np.abs(np.diff(height_lower, n=1))
        gradient_lower = np.pad(gradient_lower, (0,1), 'constant')
        
        gradient_upper = np.abs(np.diff(height_upper, n=1))
        gradient_upper = np.pad(gradient_upper, (0,1), 'constant')
        
        # Number of points to fit a line of intersection - keep this number higher than 100
        num_points = 300

        # Fetch maximum gradient points
        max_gradient_lower_points = fetch_maximum_points(X_lower, Y_lower, Z_lower, gradient_lower, num_points)
        max_gradient_upper_points = fetch_maximum_points(X_upper, Y_upper, Z_upper, gradient_upper, num_points)

        # Fit lines using PCA
        mean_lower, direction_lower = pca_line_fitting(max_gradient_lower_points)
        mean_upper, direction_upper = pca_line_fitting(max_gradient_upper_points)

        # Convert lines into open3D lines 
        line_set_lower, point1_lower, point2_lower = line_to_o3d_line(line_set_lower, mean_lower, direction_lower, length=0.5, color=[0,1,0])
        line_set_upper, point1_upper, point2_upper = line_to_o3d_line(line_set_upper, mean_upper, direction_upper, length=0.5, color=[0,0,1])

        # Calculate line equations for ROI later on
        m1, x1, y1 = calculate_line_equation(point1_lower, point2_lower)
        m2, x2, y2 = calculate_line_equation(point1_upper, point2_upper)

        # Data format to save as json (rounded to 3 decimal points)
        dataToSave = {
            "a" : round(a, 3),
            "b" : round(b, 3),
            "c" : round(c, 3),
            "d" : round(d, 3),
            "theta_x [RAD]" : round(theta_x, 3),
            "theta_y [RAD]" : round(theta_y, 3),
            "theta_x [DEG]" : round(math.degrees(theta_x), 3),
            "theta_y [DEG]" : round(math.degrees(theta_y), 3),
            "lower_line_slope" : round(m1,3),
            "upper_line_slope" : round(m2,3),
            "x1" : x1,
            "y1" : y1,
            "x2" : x2,
            "y2" : y2
        }

        # Save to JSON file
        with open("plane_model.json", "w") as json_file:
            json.dump(dataToSave, json_file, indent=4)
        
        print("[INFO] Data saved to plane_model.json.")

        # Update visualizer and its point clouds
        vis.update_geometry(pcd_inliers_vis)
        vis.update_geometry(line_set_lower)
        vis.update_geometry(line_set_upper)
        vis.poll_events()
        vis.update_renderer()
        vis.run() # Uncomment this for a frozen frame

        key_press = input("[ACTION REQUIRED] Press 'r' to fetch again or Enter 'q' to exit the code. \n")
        if key_press == 'r':
            # Close visualizer window
            vis.destroy_window()
        elif key_press == 'q':
            print("[INFO] Code terminated.")
            break 



# Terminate the code with Ctrl + C   