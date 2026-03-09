import cv2
import depthai as dai
import numpy as np
import open3d as o3d

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline = dai.Pipeline()

# RGB
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Mono
mono_left  = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Stereo depth – aligned to RGB
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(640, 480)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# XLink outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# ── Open3D visualiser ─────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("OAK-D  –  RGB Point Cloud", width=1024, height=768)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Render options
opt = vis.get_render_option()
opt.point_size        = 2.0
opt.background_color  = np.array([0.05, 0.05, 0.05])

# ── Camera intrinsics (OAK-D default for 640×480 RGB preview) ─────────────────
# You can replace these with values from device.readCalibration() if you like.
W, H = 640, 480
FX   = 457.0          # focal length x  (pixels)
FY   = 457.0          # focal length y  (pixels)
CX   = W / 2.0        # principal point x
CY   = H / 2.0        # principal point y

intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, FX, FY, CX, CY)

# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_to_colored_pcd(rgb_bgr: np.ndarray,
                          depth_mm: np.ndarray,
                          intrinsics: o3d.camera.PinholeCameraIntrinsic,
                          max_depth_m: float = 6.0,
                          depth_scale: float = 1000.0) -> o3d.geometry.PointCloud:
    """Convert an aligned (RGB, depth) pair to a coloured Open3D PointCloud."""
    # Flip vertically: OpenCV Y-axis points down, Open3D expects Y-axis up
    rgb_flipped   = cv2.flip(rgb_bgr,  0)
    depth_flipped = np.flipud(depth_mm)
    rgb_o3d   = o3d.geometry.Image(cv2.cvtColor(rgb_flipped, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_flipped.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        depth_scale=depth_scale,          # mm → m
        depth_trunc=max_depth_m,          # discard beyond this distance
        convert_rgb_to_intensity=False
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)


def colorize_by_height(pcd: o3d.geometry.PointCloud,
                        colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Per-point colors mapped to Y (height). Low=cool, High=warm."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return np.zeros((0, 3))
    y = pts[:, 1]
    y_norm = (y - y.min()) / (y.ptp() + 1e-6)
    lut = (y_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(lut.reshape(-1, 1), colormap)  # (N,1,3) BGR
    return colored.reshape(-1, 3)[:, ::-1] / 255.0             # RGB float


COLOR_MODE_RGB    = 0
COLOR_MODE_HEIGHT = 1
color_mode        = COLOR_MODE_HEIGHT   # start in height mode
first_frame       = True

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Controls (Open3D window):  Left-drag=rotate  Right-drag=pan  Scroll=zoom  R=reset")
print("Controls (cv2 window):     C=toggle RGB/Height coloring   Q/ESC=quit")

with dai.Device(pipeline) as device:

    # Optionally read factory calibration and override intrinsics
    try:
        calib   = device.readCalibration()
        M_rgb   = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, W, H)
        FX, FY  = M_rgb[0][0], M_rgb[1][1]
        CX, CY  = M_rgb[0][2], M_rgb[1][2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, FX, FY, CX, CY)
        print(f"Using device calibration  fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")
    except Exception as e:
        print(f"Could not read calibration ({e}), using defaults.")

    q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        rgb_frame   = q_rgb.get().getCvFrame()        # (H,W,3)  uint8  BGR
        depth_frame = q_depth.get().getFrame()        # (H,W)    uint16 mm

        # Build coloured point cloud
        new_pcd = depth_to_colored_pcd(rgb_frame, depth_frame, intrinsics)

        # Optional: voxel-downsample for performance (comment out if you want full res)
        new_pcd = new_pcd.voxel_down_sample(voxel_size=0.01)

        # Update geometry in-place (avoids adding duplicate geometry)
        pcd.points = new_pcd.points
        if color_mode == COLOR_MODE_HEIGHT:
            pcd.colors = o3d.utility.Vector3dVector(colorize_by_height(new_pcd))
        else:
            pcd.colors = new_pcd.colors

        if first_frame:
            vis.reset_view_point(True)
            first_frame = False

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # cv2 reference window
        depth_colored = cv2.applyColorMap(
            np.clip(depth_frame / 10000.0 * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )
        preview = np.hstack([rgb_frame, depth_colored])
        mode_label = "Height" if color_mode == COLOR_MODE_HEIGHT else "RGB"
        cv2.putText(preview, f"3D color: {mode_label}  [C to toggle]",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("RGB  |  Depth (reference)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            color_mode = COLOR_MODE_RGB if color_mode == COLOR_MODE_HEIGHT else COLOR_MODE_HEIGHT
            print(f"Color mode: {'Height' if color_mode == COLOR_MODE_HEIGHT else 'RGB'}")

vis.destroy_window()
cv2.destroyAllWindows()