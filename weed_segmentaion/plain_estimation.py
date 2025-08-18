# Imports 
import numpy as np
import depthai as dai
import cv2

# Function to get stereo pair
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

# ===== Main code ===== #

# Create a pipeline
pipeline = dai.Pipeline()

# Create and set up left and right mono cameras
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# Set properties for left camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# Set properties for right camera
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Create and set up stereo
stereo = getStereoPair(pipeline, monoLeft, monoRight)

# XLinkOut nodes - outputs
xOutDisp = pipeline.createXLinkOut()
xOutDisp.setStreamName("Disparity")

xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("Depth")

# Link Stereo and XLinkOut nodes
stereo.disparity.link(xOutDisp.input)
stereo.depth.link(xOutDepth.input)

# Focal lengths 
fx , fy = 

# Camera principle points 
cx , cy =


# Connection to OAK device
with dai.Device(pipeline) as device:
    disparityQueue = device.getOutputQueue(name="Disparity", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)
    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    # Main loop
    while True:
        disparity = disparityQueue.get()
        disparityFrame = disparity.getFrame()
        depth = depthQueue.get()

        disparityFrame = (disparityFrame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)        
        disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)
        cv2.imshow("Disparity", disparityFrame)

        # Keyboard input checking
        if cv2.waitKey(1) == ord('q'):
            break