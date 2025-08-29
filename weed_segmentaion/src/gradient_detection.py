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

    """
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
    """
    
    # Maximum gradient point lower - intersection of ground and slope plane
    max_grad_pt_lower_pcd = o3d.geometry.PointCloud()
    max_grad_pt_lower_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    max_grad_pt_lower_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    vis.add_geometry(max_grad_pt_lower_pcd)

    # Maximum gradient point upper - intersection of ground and slope plane
    max_grad_pt_upper_pcd = o3d.geometry.PointCloud()
    max_grad_pt_upper_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
    max_grad_pt_upper_pcd.paint_uniform_color([1.0, 1.0, 1.0])  # Black
    vis.add_geometry(max_grad_pt_upper_pcd)

    # Green line (LineSet) parallel to X-axis 
    line_set_lower = o3d.geometry.LineSet()
    line_set_lower.points = o3d.utility.Vector3dVector(np.zeros((2,3)))  # 2,3 - Two points in coordinate center initially  [0, 0, 0] and [0, 0, 0]                                                                                                                       [0, 0, 0]
    line_set_lower.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set_lower.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
    vis.add_geometry(line_set_lower)

    # Green line (LineSet) parallel to X-axis 
    line_set_upper = o3d.geometry.LineSet()
    line_set_upper.points = o3d.utility.Vector3dVector(np.zeros((2,3))) 
    line_set_upper.lines = o3d.utility.Vector2iVector([[0,1]])
    line_set_upper.colors = o3d.utility.Vector3dVector([[0.3, 0.15, 0]])  # Brown 
    vis.add_geometry(line_set_upper)

    while True:
        # Fetch a depth frame (metrics in mm) and convert it for precission
        depth = depthQueue.get()
        depthFrame = depth.getFrame()#.astype(np.float32) 
        
        # From mm to m
        Z = depthFrame / 1000.0 
        
        ## Noise removal with median filtering
        depthFrame = cv2.medianBlur(depthFrame, 5)

        ## ================================================= ##

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
        
        # Extract Carthesian parameter values
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

        # Compute gradients for both sections
        gradient_lower = np.abs(np.diff(height_lower, n=1))
        gradient_lower = np.pad(gradient_lower, (0,1), 'constant')

        gradient_upper = np.abs(np.diff(height_upper, n=1))
        gradient_upper = np.pad(gradient_upper, (0,1), 'constant')

        # Maximum gradient values
        max_gradient_lower = np.argmax(gradient_lower)
        max_gradient_upper = np.argmax(gradient_upper)

        # Define points for these maximums
        max_gradient_lower_point = np.array([X_lower[max_gradient_lower], Y_lower[max_gradient_lower], Z_lower[max_gradient_lower]])
        max_gradient_upper_point = np.array([X_upper[max_gradient_upper], Y_upper[max_gradient_upper], Z_upper[max_gradient_upper]])
        
        # Open3D points for maximum gradient values
        max_grad_pt_lower_pcd.points = o3d.utility.Vector3dVector(max_gradient_lower_point.reshape(1,3))
        max_grad_pt_upper_pcd.points = o3d.utility.Vector3dVector(max_gradient_upper_point.reshape(1,3))

        # Green line along X-axis
        x_range = np.linspace(-1.0, 1.0, 50)  # Draw a 50 point line from -1 to 1 (ground plane parallel) 
        
        # Fixed values for maximum gradient point 
        y_fixed_lower = max_gradient_lower_point[1]
        y_fixed_higher = max_gradient_upper_point[1]

        z_fixed_lower = max_gradient_lower_point[2]
        z_fixed_higher = max_gradient_upper_point[2]
        
        # Combination of 3 coordinates to create a 3D space for the line
        # full_like fills y_fixed and z_fixed to be the same dimension as x_range
        line_points_lower = np.stack([x_range, np.full_like(x_range, y_fixed_lower), np.full_like(x_range, z_fixed_lower)], axis=1)
        line_points_upper = np.stack([x_range, np.full_like(x_range, y_fixed_higher), np.full_like(x_range, z_fixed_higher)], axis=1)

        # Line parameters and configuration - lower frame
        line_set_lower.points = o3d.utility.Vector3dVector(line_points_lower)
        lines_lower = [[i, i+1] for i in range(len(line_points_lower)-1)]
        line_set_lower.lines = o3d.utility.Vector2iVector(lines_lower)
        line_set_lower.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(lines_lower),1)))

        # Line parameters and configuration - upper frame
        line_set_upper.points = o3d.utility.Vector3dVector(line_points_upper)
        lines_upper = [[i, i+1] for i in range(len(line_points_upper)-1)]
        line_set_upper.lines = o3d.utility.Vector2iVector(lines_upper)
        line_set_upper.colors = o3d.utility.Vector3dVector(np.tile([0.3,0.15,0], (len(lines_upper),1)))


        # ===================================================================== #

        """

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

        """

        # Update visualizer and its point clouds
        vis.update_geometry(pcd_inliers_vis)
        #vis.update_geometry(max_pcd)
        #vis.update_geometry(line_set)
        vis.update_geometry(max_grad_pt_lower_pcd)
        vis.update_geometry(max_grad_pt_upper_pcd)
        vis.update_geometry(line_set_lower)
        vis.update_geometry(line_set_upper)
        vis.poll_events()
        vis.update_renderer()


# Terminate the code with Ctrl + C