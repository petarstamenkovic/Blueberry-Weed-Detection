# This code detects blueberry weed on bank and outputs necessary coordinates for further processing

# Import libraries 
import depthai as dai
import cv2
import numpy as np

# Create a pipeline 
pipeline = dai.Pipeline()

# Create camera nodes - RGB and MONO 
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# Cameras configuration
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Create stereo node 
stereoDepth = pipeline.create(dai.node.StereoDepth)
stereoDepth.initialConfig.setConfidenceThreshold(255)
stereoDepth.setLeftRightCheck(True)
stereoDepth.setSubpixel(False)

# Link mono cameras to a stereo depth node
monoLeft.out.link(stereoDepth.left)
monoRight.out.link(stereoDepth.right)

# Create XLinkOut node 
xOutDepth = pipeline.create(dai.node.XLinkOut)
xOutDepth.setStreamName("Depth")

# Link outputs of nodes to XLinkOut
stereoDepth.depth.link(xOutDepth.input)

# Main loop
with dai.Device(pipeline) as device:
    # Create output queues
    depthQueue = device.getOutputQueue(name="Depth", maxSize=4, blocking=False)
    camQueue = device.getOutputQueue(name="Camera", maxSize=4, blocking=False)

    while True:
        # Get frames from the queues
        depthFrame = depthQueue.get()
        camFrame = camQueue.get()
        
        # Process the camera frame
        frame = cv2.cvtColor(camFrame.getCvFrame(), cv2.COLOR_BGR2RGB)

        # Here you would add your detection logic for blueberry weed
        # For example, using a pre-trained model or simple color thresholding

        # Display the results
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break



