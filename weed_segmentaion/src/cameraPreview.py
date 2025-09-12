# This code provides RGB camera preview using DepthAI. Mainly used to setup a good camera angle since main code is depth view with point cloud.

# Necessary libraries
import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Create a RGB camera node - Just for preview and camera postion setup
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# XLinkOut node for RGB camera
xOutCamRgb = pipeline.createXLinkOut()
xOutCamRgb.setStreamName("CamRgb")

# Link RGB camera to XLinkOut node
camRgb.video.link(xOutCamRgb.input)

# Connection to OAK device
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="CamRgb", maxSize=4, blocking=False)

    # Main loop
    while True:
        # RGB frame fetch
        rgb = rgbQueue.get()
        rgbFrame = rgb.getCvFrame()

        cv2.imshow("RGB", rgbFrame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()