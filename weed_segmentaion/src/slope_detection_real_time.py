# This code uses gradient calculation and point cloud differences to find the lines of intersection

# Necessary imports
import json
import math
import time
import numpy as np
import depthai as dai
import open3d as o3d

# Parameters for plane comparison
normal_vector_threshold = 0.5
offset_threshold = 0.5

# Camera intrinsics - Extracted from calibration.py
#fx, fy = 570.33, 570.3   # Focal lenghts
#cx, cy = 655.38, 358.85  # Camera principle values 

# Parameters for OAK D - Camera number 9
fx = 856.8466
fy = 857.0558
cx = 632.1884
cy = 366.1514


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

# Function that tries to maintain the focused plain
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

    # Comparison of the planes under conditions
    if normal_vectors_alligned > (1 - normal_vector_threshold) and offset_diffence < offset_threshold:
        return newPlane  # Update the plane with a new one 
    else:
        return prevPlane # Keep the previous plane

# Fetch first num_points number of maximum gradient points
def fetchMaximumPoints(X, Y, Z, gradient_array, num_points):
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
def pcaLineFitting(points):

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
def backtrackTo2D(endpoint):
    X, Y, Z = endpoint
    
    # Skip if Z is 0 due to division
    if Z == 0:
        return None
    
    # Backtrack - Both u and v are in pixel values - these will limit the ROI 
    u = int((X * fx) / Z + cx)
    v = int((Y * fy) / Z + cy)

    return (u,v)

# Convert PCA lines to Open3D lines
def lineToO3dLine(line_set, mean_point, direction, length=0.5, color=[0,1,0]):
    direction = direction / np.linalg.norm(direction)

    # Create endpoints of the line segment
    endpoint1 = mean_point - (direction * length / 2)
    endpoint2 = mean_point + (direction * length / 2)

    """
    print("Endpoint 1 : ", endpoint1)
    print("-------------------------------------")
    print("Endpoint 2 : ", endpoint2)
    print("=====================================")
    """

    # Define line parameters
    line_points = [endpoint1, endpoint2] # Actual points of the line
    lines = [[0, 1]]  # Single line connecting the two points - index based ; "connect 0-th and 1-st point"

    # Points in 2D - For line equation and ROI configuration later in the process
    point1 = backtrackTo2D(endpoint1)
    point2 = backtrackTo2D(endpoint2)

    # Add line parameters
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set, point1, point2

# Function that calculates 2D line equation
def calculateLineEquation(point1, point2):
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
 
    # Initialize time stamp
    last_write_time = time.time()

    # Main loop for processing
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
            distance_threshold=0.02, # Try adjusting this parameter in case plane starts shifting
            ransac_n=3, 
            num_iterations=1000) # Try calculating it with a given formula
 
        # Function that keeps a slope plane locked - Deactivated
        #activePlane = updatePlane(previousPlane, plane_model)

        # Extract Cartesian parameter values
        a, b, c, d = plane_model
        #previousPlane = activePlane

        # Extract inliers and prepare them for visualization 
        pcd_inliers = pcd.select_by_index(inliers)

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

        # Not necessary, data is written in json, use this for debugging in terminal
        """
        print("=====================================================")
        print(f"Tilt around x axis in RAD is {theta_x} and in DEG is {round(math.degrees(theta_x),2)}.")
        print(f"Tilt around y axis in RAD is {theta_y} and in DEG is {round(math.degrees(theta_y),2)}.")
        print("======================================================")
        """
        # =============================================================== #

        # DBScan clustering - Extremely heavy on the visualizer that it becomes inresponsive
        #labels = np.array(
        #    pcd_inliers.cluster_dbscan(eps = 0.05, min_points = 10)
        #)
        #max_label = labels.max()
    
        # Configure inliers visualization
        pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_inliers_vis.points = pcd_inliers.points
        pcd_inliers_vis.colors = pcd_inliers.colors

        # X_in - Real-life x coordinates of inlier points
        # Y_in - Real-life y coordinates of inlier points
        # Z_in - Distance from the camera of inlier points
        X_in = np.asarray(pcd_inliers.points)[:,0]
        Y_in = np.asarray(pcd_inliers.points)[:,1]
        Z_in = np.asarray(pcd_inliers.points)[:,2]

        # How much is the point far away from the ideal segmented plane 
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
        num_points = 300

        # Fetch maximum gradient points
        max_gradient_lower_points = fetchMaximumPoints(X_lower, Y_lower, Z_lower, gradient_lower, num_points)
        max_gradient_upper_points = fetchMaximumPoints(X_upper, Y_upper, Z_upper, gradient_upper, num_points)

        # Fit lines using PCA
        mean_lower, direction_lower = pcaLineFitting(max_gradient_lower_points)
        mean_upper, direction_upper = pcaLineFitting(max_gradient_upper_points)

        # Convert lines into open3D lines 
        line_set_lower, point1_lower, point2_lower = lineToO3dLine(line_set_lower, mean_lower, direction_lower, length=0.5, color=[0,1,0])
        line_set_upper, point1_upper, point2_upper = lineToO3dLine(line_set_upper, mean_upper, direction_upper, length=0.5, color=[0,0,1])

        # Calculate line equations for ROI later on
        m1, x1, y1 = calculateLineEquation(point1_lower, point2_lower)
        m2, x2, y2 = calculateLineEquation(point1_upper, point2_upper)

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

        # Save the plane model to a JSON file every 3 seconds
        current_time = time.time()
        if current_time - last_write_time >= 3:
            with open("plane_model.json", "w") as json_file:
                json.dump(dataToSave, json_file, indent=4)
            last_write_time = current_time

        # Not necessary, data is written in json file
        """
        print("======================================================")
        print(f"Lower line of intersection equation : (y - {y1}) = {m1} * (x - {x1})")
        print(f"Upper line of intersection equation : (y - {y2}) = {m2} * (x - {x2})")
        print("======================================================")
        """

        # Update visualizer and its point clouds
        vis.update_geometry(pcd_inliers_vis)
        vis.update_geometry(line_set_lower)
        vis.update_geometry(line_set_upper)
        vis.get_render_option().line_width = 10
        vis.poll_events()
        vis.update_renderer()
        #vis.run() # Uncomment this for a frozen frame

# Terminate the code with Ctrl + C