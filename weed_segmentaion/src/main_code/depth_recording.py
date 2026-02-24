import depthai as dai
import numpy as np
import cv2
import os
import utils

NUM_FRAMES = 200
OUTPUT_DIR = "oak_depth_data_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build a pipeline
pipeline = utils.dai.Pipeline()
stereo = utils.buildStereoPipeline(pipeline)

# XLinkOut node for depth output
xOutDepth = pipeline.createXLinkOut()
xOutDepth.setStreamName("depth")
stereo.depth.link(xOutDepth.input)

with dai.Device(pipeline) as device:

    # Save intrinsics
    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.LEFT, 640, 400
    )
    np.save(os.path.join(OUTPUT_DIR, "intrinsics.npy"), np.array(intrinsics))

    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

    for i in range(NUM_FRAMES):
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()

        np.save(
            os.path.join(OUTPUT_DIR, f"depth4_{i:05d}.npy"),
            depthFrame
        )

        print(f"Saved {i+1}/{NUM_FRAMES}")

print("Done.")