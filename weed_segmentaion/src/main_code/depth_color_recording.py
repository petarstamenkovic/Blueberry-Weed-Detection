import depthai as dai
import numpy as np
import cv2
import os
import csv
import math

# --- Change this before each run ---
SCENE_ID = 1  # 0, 1, 2, 3, 4, 5

# --- Output folders ---
OUTPUT_DIR = f"scene_{SCENE_ID}"
COLOR_DIR = os.path.join(OUTPUT_DIR, "color")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
for d in [COLOR_DIR, DEPTH_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Pipeline ---
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

imu = pipeline.create(dai.node.IMU)
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 100)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_imu = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
xout_imu.setStreamName("imu")

cam_rgb.video.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)
imu.out.link(xout_imu.input)

def quat_to_euler(qw, qx, qy, qz):
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
    return roll, pitch, yaw

frame_idx = 0
latest_euler = (0.0, 0.0, 0.0)

with dai.Device(pipeline) as device, open(os.path.join(OUTPUT_DIR, "imu_log.csv"), "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["frame_idx", "roll_deg", "pitch_deg", "yaw_deg"])

    q_rgb   = device.getOutputQueue("rgb",   maxSize=4,  blocking=True)
    q_depth = device.getOutputQueue("depth", maxSize=4,  blocking=True)
    q_imu   = device.getOutputQueue("imu",   maxSize=50, blocking=False)

    while True:
        imu_data = q_imu.tryGet()
        if imu_data:
            for packet in imu_data.packets:
                rv = packet.rotationVector
                latest_euler = quat_to_euler(rv.real, rv.i, rv.j, rv.k)

        roll, pitch, yaw = latest_euler

        in_rgb   = q_rgb.get()
        in_depth = q_depth.get()

        color_frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        dh, dw = depth_frame.shape
        color_resized = cv2.resize(color_frame, (dw, dh))

        cv2.imwrite(os.path.join(COLOR_DIR, f"frame_{frame_idx:05d}.png"), color_resized)
        cv2.imwrite(os.path.join(DEPTH_DIR, f"frame_{frame_idx:05d}.png"), depth_frame.astype(np.uint16))
        writer.writerow([frame_idx, round(roll, 2), round(pitch, 2), round(yaw, 2)])

        cv2.putText(color_resized, f"Pitch: {pitch:.1f}  Roll: {roll:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("Color", color_resized)
        cv2.imshow("Depth", cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

        print(f"Frame {frame_idx:05d} | Pitch: {pitch:.1f}°  Roll: {roll:.1f}°", end="\r")
        frame_idx += 1

        if cv2.waitKey(1) == ord('q'):
            print(f"\nDone. {frame_idx} frames saved to {OUTPUT_DIR}/")
            break

cv2.destroyAllWindows()