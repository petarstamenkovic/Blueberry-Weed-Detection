# This code attempts to repeat the algorithm in an offline (not real time) manner. After algorithm is good 
# and precise, try to revert to this code for speed optimization.

# Necessary libraries 
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

# Function that tries to focus the correct plane
def updatePlane(prevPlane, newPlane):
    a,b,c,d = newPlane
    new_normal_vector = np.array([a,b,c])
    new_normal_vector = new_normal_vector / np.linalg.norm(new_normal_vector)
    if prevPlane is None:
        return newPlane
    prev_a, prev_b, prev_c, prev_d = prevPlane
    old_normal_vector = np.array([prev_a, prev_b, prev_c])
    old_normal_vector = old_normal_vector / np.linalg.norm(old_normal_vector)
    normal_vectors_alligned = abs(np.dot(new_normal_vector, old_normal_vector))
    offset_diffence = abs(d - prev_d)
    if normal_vectors_alligned > (1 - normal_vector_threshold) and offset_diffence < offset_threshold:
        return newPlane
    else:
        return prevPlane

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

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live stream", width=1920, height=1080, visible=False)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

# Plane inliers (slope)
pcd_inliers_vis = o3d.geometry.PointCloud()
pcd_inliers_vis.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
pcd_inliers_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red
vis.add_geometry(pcd_inliers_vis)

# Max gradient points
max_grad_pt_lower_pcd = o3d.geometry.PointCloud()
max_grad_pt_lower_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
max_grad_pt_lower_pcd.paint_uniform_color([1.0, 1.0, 0.0])
vis.add_geometry(max_grad_pt_lower_pcd)

max_grad_pt_upper_pcd = o3d.geometry.PointCloud()
max_grad_pt_upper_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
max_grad_pt_upper_pcd.paint_uniform_color([1.0, 1.0, 1.0])
vis.add_geometry(max_grad_pt_upper_pcd)

# Green lines
line_set_lower = o3d.geometry.LineSet()
line_set_lower.points = o3d.utility.Vector3dVector(np.zeros((2,3)))
line_set_lower.lines = o3d.utility.Vector2iVector([[0,1]])
line_set_lower.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
vis.add_geometry(line_set_lower)

line_set_upper = o3d.geometry.LineSet()
line_set_upper.points = o3d.utility.Vector3dVector(np.zeros((2,3)))
line_set_upper.lines = o3d.utility.Vector2iVector([[0,1]])
line_set_upper.colors = o3d.utility.Vector3dVector([[0.3, 0.15, 0]])
vis.add_geometry(line_set_upper)

# Connect to device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)

    # Minimal OpenCV window for keypress detection
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth", 1,1)
    cv2.moveWindow("Depth", -100,-100)

    while True:
        if cv2.waitKey(1) & 0xFF == ord(' '):
            depth = depthQueue.get()
            depthFrame = depth.getFrame()
            Z = depthFrame / 1000.0
            depthFrame = cv2.medianBlur(depthFrame, 5)
            h, w = Z.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0)
            pts = pts[valid]

            if pts.shape[0] < 10:
                print("[INFO] Not enough valid points found... skipping frame")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            activePlane = updatePlane(previousPlane, plane_model)
            a, b, c, d = activePlane
            previousPlane = activePlane
            pcd_inliers = pcd.select_by_index(inliers)

            pcd_inliers.paint_uniform_color([1.0, 0.0, 0.0])
            pcd_inliers_vis.points = pcd_inliers.points
            pcd_inliers_vis.colors = pcd_inliers.colors

            # Gradient computation
            X_in = np.asarray(pcd_inliers.points)[:,0]
            Y_in = np.asarray(pcd_inliers.points)[:,1]
            Z_in = np.asarray(pcd_inliers.points)[:,2]
            height_values = a*X_in + b*Y_in + c*Z_in + d

            y_threshold = np.median(Y_in)
            lower_mask = Y_in < y_threshold
            upper_mask = Y_in > y_threshold
            X_lower, Y_lower, Z_lower = X_in[lower_mask], Y_in[lower_mask], Z_in[lower_mask]
            X_upper, Y_upper, Z_upper = X_in[upper_mask], Y_in[upper_mask], Z_in[upper_mask]

            height_lower = height_values[lower_mask]
            height_upper = height_values[upper_mask]
            gradient_lower = np.pad(np.abs(np.diff(height_lower)), (0,1), 'constant')
            gradient_upper = np.pad(np.abs(np.diff(height_upper)), (0,1), 'constant')
            max_gradient_lower_point = np.array([X_lower[np.argmax(gradient_lower)],
                                                 Y_lower[np.argmax(gradient_lower)],
                                                 Z_lower[np.argmax(gradient_lower)]])
            max_gradient_upper_point = np.array([X_upper[np.argmax(gradient_upper)],
                                                 Y_upper[np.argmax(gradient_upper)],
                                                 Z_upper[np.argmax(gradient_upper)]])

            max_grad_pt_lower_pcd.points = o3d.utility.Vector3dVector(max_gradient_lower_point.reshape(1,3))
            max_grad_pt_upper_pcd.points = o3d.utility.Vector3dVector(max_gradient_upper_point.reshape(1,3))

            # Green lines
            x_range = np.linspace(-1.0, 1.0, 50)
            line_set_lower.points = o3d.utility.Vector3dVector(
                np.stack([x_range, np.full_like(x_range, max_gradient_lower_point[1]),
                          np.full_like(x_range, max_gradient_lower_point[2])], axis=1))
            line_set_upper.points = o3d.utility.Vector3dVector(
                np.stack([x_range, np.full_like(x_range, max_gradient_upper_point[1]),
                          np.full_like(x_range, max_gradient_upper_point[2])], axis=1))
            lines_lower = [[i,i+1] for i in range(len(x_range)-1)]
            line_set_lower.lines = o3d.utility.Vector2iVector(lines_lower)
            line_set_lower.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(lines_lower),1)))
            lines_upper = [[i,i+1] for i in range(len(x_range)-1)]
            line_set_upper.lines = o3d.utility.Vector2iVector(lines_upper)
            line_set_upper.colors = o3d.utility.Vector3dVector(np.tile([0.3,0.15,0], (len(lines_upper),1)))

            # --- Auto adjust view control to slope center --- #
            ctr = vis.get_view_control()
            if len(pcd_inliers.points) > 0:
                lookat_center = np.mean(np.asarray(pcd_inliers.points), axis=0)
                ctr.set_lookat(lookat_center)
                ctr.set_up([0,1,0])
                ctr.set_front([0, -0.3, -1.0])  # Slightly above looking downward
                ctr.set_zoom(0.6)

            # Update visualizer and capture
            vis.update_geometry(pcd_inliers_vis)
            vis.update_geometry(max_grad_pt_lower_pcd)
            vis.update_geometry(max_grad_pt_upper_pcd)
            vis.update_geometry(line_set_lower)
            vis.update_geometry(line_set_upper)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("gradient_snapshot.png", do_render=True)
            print("[INFO] Snapshot saved as 'gradient_snapshot.png'")

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Terminating the code...")
            break
