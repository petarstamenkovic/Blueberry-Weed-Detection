# Python script for plane estimation using DepthAI stereo cameras

# Imports 
import numpy as np
import depthai as dai
import cv2
import open3d as o3d
import math
import json
import time

# Function to get stereo pair
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

# ===== Main code ===== #

# Create a pipeline
pipeline = dai.Pipeline()

# Create a RGB camera node - Just for preview and camera postion setup
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create and set up left and right mono cameras
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# Set properties for left camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# Set properties for right camera
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Create and set up stereo
stereo = getStereoPair(pipeline, monoLeft, monoRight)

# XLinkOut nodes - outputs
xOutDisp = pipeline.createXLinkOut()
xOutDisp.setStreamName("Disparity")

xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("Depth")

xOutCamRgb = pipeline.createXLinkOut()
xOutCamRgb.setStreamName("CamRgb")

# Link Stereo and XLinkOut nodes
stereo.disparity.link(xOutDisp.input)
stereo.depth.link(xOutDepth.input)

# Link RGB camera to XLinkOut node
camRgb.video.link(xOutCamRgb.input)

# Focal lengths and camera principle points - Extracted from calibration.py
fx , fy = 570.33, 570.300
cx , cy = 655.38, 358.85

# Conversion from 2D depth to 3D points cloud format 
def depth_to_points(depth_mm, fx, fy, cx, cy):
    # depth in millimeters -> meters
    D = depth_mm.astype(np.float32) / 1000.0
    h, w = D.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    # Back-project
    X = (uu - cx) * D / fx
    Y = (vv - cy) * D / fy
    Z = D
    # Stack valid points
    valid = (D > 0) & np.isfinite(D)
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    return pts


# Connection to OAK device
with dai.Device(pipeline) as device:
    disparityQueue = device.getOutputQueue(name="Disparity", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue(name="CamRgb", maxSize=4, blocking=False)

    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    last_write_time = time.time()

    # Main loop
    while True:
        # Disparity frame fetch
        disparity = disparityQueue.get()
        disparityFrame = disparity.getFrame()

        # Depth frame fetch
        depth = depthQueue.get()
        depthFrame = depth.getFrame()

        # RGB frame fetch
        rgb = rgbQueue.get()
        rgbFrame = rgb.getCvFrame()

        # ROI Configuration - Ignore the upper top quarter of the frame
        h,w = depthFrame.shape
        roi = depthFrame[h//4:h , :]
        #pts = depth_to_points(roi, fx, fy, cx, cy - h // 4)
        pts = depth_to_points(depthFrame, fx, fy, cx, cy)
        #print("Points : ", pts.shape[0])

        # First couple of frames upon waking the camera up, depth is not stable
        if pts.shape[0] < 3:
            print("[INFO] Not enought valid points found... skipping frame")
            continue
        else:
            # Reduce the number of points for RANSAC
            limit = 200000 # Adjust this number if necessary
            if pts.shape[0] > limit: 
                idx = np.random.choice(pts.shape[0], limit, replace=False)
                pts = pts[idx]

            # Integrate open3D library - pcd as point cloud data
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            # RANSAC Plane fitting - Returns plane parameters from plane equation
            # ax + by + cz + d = 0
            plane_model, inliers = pcd.segment_plane(
                distance_threshold = 0.01, # 1cm - Adjust this (increase) if fitting fails
                ransac_n = 3,
                num_iterations = 1000
            )

            a,b,c,d = plane_model
            n = np.array([a,b,c], dtype=np.float64)
            n = n / np.linalg.norm(n)

            # Make normal point UP since camera y axis is DOWN
            if n[1] > 0:
                n = -n

            # Extract angles
            theta_x = math.atan2(n[2], -n[1])
            theta_y = math.atan2(n[0], -n[1])

            # Data format to save as json (rounded to 3 decimal points)
            dataToSave = {
                "a" : round(a, 3),
                "b" : round(b, 3),
                "c" : round(c, 3),
                "d" : round(d, 3),
                "theta_x [RAD]" : round(theta_x, 3),
                "theta_y [RAD]" : round(theta_y, 3),
                "theta_x [DEG]" : round(math.degrees(theta_x), 3),
                "theta_y [DEG]" : round(math.degrees(theta_y), 3)
            }

            # Save the plane model to a JSON file every 3 seconds
            current_time = time.time()
            if current_time - last_write_time >= 3:
                with open("plane_model.json", "w") as json_file:
                    json.dump(dataToSave, json_file, indent=4)
                last_write_time = current_time

            print("Plane: ax+by+cz+d=0 ->", plane_model)
            #print("Unit normal n:", n)
            print("Tilt about X (rad, deg):", theta_x, math.degrees(theta_x))
            print("Tilt about Y (rad, deg):", theta_y, math.degrees(theta_y))

            #disparityFrame = (disparityFrame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)        
            #disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)
            #cv2.imshow("Disparity", disparityFrame)
