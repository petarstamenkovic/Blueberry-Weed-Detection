import cv2
import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Output to host
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        rgb_frame = q_rgb.get().getCvFrame()
        
        # Show RGB frame
        cv2.imshow("RGB Preview", rgb_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

cv2.destroyAllWindows()