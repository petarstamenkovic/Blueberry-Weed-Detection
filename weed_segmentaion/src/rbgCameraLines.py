import numpy as np
import depthai as dai
import cv2

# ========================= PARAMETERS ========================= #
# Plane update thresholds
normal_vector_threshold = 0.1
offset_threshold = 0.1

# Camera intrinsics (from calibration)
fx, fy = 570.33, 570.3
cx, cy = 655.38, 358.85

# Number of gradient points to fit the line
num_points = 50

# ========================= FUNCTIONS ========================= #
def updatePlane(prevPlane, newPlane):
    a,b,c,d = newPlane
    new_normal = np.array([a,b,c])
    new_normal /= np.linalg.norm(new_normal)

    if prevPlane is None:
        return newPlane

    prev_a, prev_b, prev_c, prev_d = prevPlane
    old_normal = np.array([prev_a, prev_b, prev_c])
    old_normal /= np.linalg.norm(old_normal)

    normal_aligned = abs(np.dot(new_normal, old_normal))
    offset_diff = abs(d - prev_d)

    if normal_aligned > (1 - normal_vector_threshold) and offset_diff < offset_threshold:
        return newPlane
    else:
        return prevPlane

def fetch_max_gradient_points(X, Y, Z, gradient_array, n_points):
    top_idx = np.argsort(gradient_array)[::-1][:n_points]
    return np.stack([X[top_idx], Y[top_idx], Z[top_idx]], axis=1)

def pca_line_fitting(points):
    mean_point = np.mean(points, axis=0)
    centered = points - mean_point
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    return mean_point, direction

def project_line_to_image(mean_point, direction, fx, fy, cx, cy, length=0.5):
    # Create endpoints in 3D
    endpoint1 = mean_point - direction*length/2
    endpoint2 = mean_point + direction*length/2

    # Project to 2D
    u1 = int(fx*endpoint1[0]/endpoint1[2] + cx)
    v1 = int(fy*endpoint1[1]/endpoint1[2] + cy)
    u2 = int(fx*endpoint2[0]/endpoint2[2] + cx)
    v2 = int(fy*endpoint2[1]/endpoint2[2] + cy)
    return (u1, v1), (u2, v2)

# ========================= CAMERA PIPELINE ========================= #
pipeline = dai.Pipeline()

# Mono cameras for depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# RGB camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1280, 720)
camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(False)

xOutRgb = pipeline.create(dai.node.XLinkOut)
xOutRgb.setStreamName("rgb")
camRgb.preview.link(xOutRgb.input)

xOutDepth = pipeline.create(dai.node.XLinkOut)
xOutDepth.setStreamName("depth")
stereo.depth.link(xOutDepth.input)

# ========================= MAIN ========================= #
previousPlane = None

with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

    # Capture a single frame
    rgbFrame = rgbQueue.get().getCvFrame()
    depthFrame = depthQueue.get().getFrame().astype(np.float32) / 1000.0  # meters

    # Back-project depth to 3D
    h, w = depthFrame.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * depthFrame / fx
    Y = (v - cy) * depthFrame / fy
    Z = depthFrame

    # Filter valid points
    pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (Z.flatten() > 0)
    pts = pts[valid]

    if pts.shape[0] < 10:
        print("Not enough points")
        exit()

    # Plane fitting via RANSAC
    pcd_points = pts
    # Fit plane using least squares: ax + by + cz + d = 0
    A = np.c_[pcd_points[:,0], pcd_points[:,1], pcd_points[:,2], np.ones(pcd_points.shape[0])]
    _, _, V = np.linalg.svd(A)
    plane = V[-1,:]
    plane = plane / np.linalg.norm(plane[:3])
    a,b,c,d = plane
    activePlane = updatePlane(previousPlane, (a,b,c,d))
    previousPlane = activePlane

    # Compute height above plane
    height_values = a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d

    # Split upper/lower by Y median
    Y_in = pts[:,1]
    median_y = np.median(Y_in)
    lower_mask = Y_in < median_y
    upper_mask = Y_in >= median_y

    X_lower, Y_lower, Z_lower = pts[lower_mask,0], pts[lower_mask,1], pts[lower_mask,2]
    X_upper, Y_upper, Z_upper = pts[upper_mask,0], pts[upper_mask,1], pts[upper_mask,2]
    height_lower = height_values[lower_mask]
    height_upper = height_values[upper_mask]

    # Sort by Y
    sort_lower = np.argsort(Y_lower)
    X_lower, Y_lower, Z_lower, height_lower = X_lower[sort_lower], Y_lower[sort_lower], Z_lower[sort_lower], height_lower[sort_lower]
    sort_upper = np.argsort(Y_upper)
    X_upper, Y_upper, Z_upper, height_upper = X_upper[sort_upper], Y_upper[sort_upper], Z_upper[sort_upper], height_upper[sort_upper]

    # Compute gradients
    grad_lower = np.abs(np.diff(height_lower, n=1))
    grad_lower = np.pad(grad_lower, (0,1), 'constant')
    grad_upper = np.abs(np.diff(height_upper, n=1))
    grad_upper = np.pad(grad_upper, (0,1), 'constant')

    # Max gradient points
    max_lower_pts = fetch_max_gradient_points(X_lower, Y_lower, Z_lower, grad_lower, num_points)
    max_upper_pts = fetch_max_gradient_points(X_upper, Y_upper, Z_upper, grad_upper, num_points)

    # PCA line fitting
    mean_lower, dir_lower = pca_line_fitting(max_lower_pts)
    mean_upper, dir_upper = pca_line_fitting(max_upper_pts)

    # Project lines onto RGB image
    (u1l,v1l),(u2l,v2l) = project_line_to_image(mean_lower, dir_lower, fx, fy, cx, cy)
    (u1u,v1u),(u2u,v2u) = project_line_to_image(mean_upper, dir_upper, fx, fy, cx, cy)

    # Draw lines
    cv2.line(rgbFrame, (u1l,v1l), (u2l,v2l), (0,255,0), 2)
    cv2.line(rgbFrame, (u1u,v1u), (u2u,v2u), (255,0,0), 2)

    # Show result and save
    cv2.imshow("RGB with Lines", rgbFrame)
    cv2.imwrite("snapshot_with_lines.png", rgbFrame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
