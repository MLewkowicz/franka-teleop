"""ZED 2 extrinsics calibration.

Eye-on-base hand-eye calibration: the ChArUco board is rigidly attached
to the gripper at an unknown offset; the third-person ZED 2 is stationary.
At each captured pose we record the end-effector pose and a synchronized
camera frame, detect the ChArUco board, then solve ``cv2.calibrateHandEye``
for ``T_cam2base``.

Eye-in-hand calibration: the ZED is rigidly attached to the gripper/wrist
and the ChArUco board is fixed in the robot base/world. The same capture
flow solves ``T_cam2gripper`` and writes both ``T_cam2gripper`` and its
inverse ``T_gripper2cam``.

Manual capture uses low Cartesian impedance + live camera preview. The
operator physically guides the arm to each pose and presses SPACE to capture.

Output: a JSON file with the recovered transform(s), intrinsics snapshot,
and quality statistics.
"""

import datetime
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from zero_franky import Robot
from zero_franky.tracker_policies import hold_current_joint

from clear_franka.camera import ZedCamera, get_camera_config, make_zed_camera
from clear_franka.geometry import (
    average_transforms as _average_transforms,
    invert_transform as _invert_transform,
    make_transform as _make_transform,
)
from clear_franka.utils import prompt_yes_no, wait_for_enter

logger = logging.getLogger(__name__)

DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


# ---------------------------------------------------------------------------
# Hand-eye solve helpers
# ---------------------------------------------------------------------------

def _solve_hand_eye(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
    """Eye-on-base: recover cam2base."""
    import cv2
    R_base2gripper = [R.T for R in R_gripper2base_list]
    t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base_list, t_gripper2base_list)]
    R_c2b, t_c2b = cv2.calibrateHandEye(
        R_gripper2base=R_base2gripper, t_gripper2base=t_base2gripper,
        R_target2cam=R_target2cam_list, t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    return _make_transform(R_c2b, t_c2b.flatten())


def _solve_eye_in_hand(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
    """Eye-in-hand: recover cam2gripper."""
    import cv2
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list, t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list, t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    return _make_transform(R_c2g, t_c2g.flatten())


def _solve_board2gripper(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list, T_cam2base):
    R_c2b, t_c2b = T_cam2base[:3, :3], T_cam2base[:3, 3]
    Ts = []
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
        R_t2g = R_g2b.T @ R_c2b @ R_t2c
        t_t2g = R_g2b.T @ (R_c2b @ t_t2c + t_c2b - t_g2b)
        Ts.append(_make_transform(R_t2g, t_t2g))
    return _average_transforms(Ts), Ts


def _solve_board2base(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list, T_cam2gripper):
    Ts = []
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
        Ts.append(_make_transform(R_g2b, t_g2b) @ T_cam2gripper @ _make_transform(R_t2c, t_t2c))
    return _average_transforms(Ts), Ts


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------


def _build_board(cfg: DictConfig):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    board = cv2.aruco.CharucoBoard(
        size=(cfg.squares_x, cfg.squares_y),
        squareLength=cfg.square_length_m,
        markerLength=cfg.marker_length_m,
        dictionary=aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def _detect_board_pose(detector, board, rgb, K, dist, min_corners):
    """Detect the ChArUco board and solve its pose in the camera frame.

    Returns a dict with R_target2cam, t_target2cam, obj_pts, img_pts,
    charuco_corners, charuco_ids, marker_corners, marker_ids, n_corners
    (the marker* fields may be None) — or None if detection / PnP failed.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < min_corners:
        return None

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R_target2cam, _ = cv2.Rodrigues(rvec)
    return {
        "R_target2cam": R_target2cam,
        "t_target2cam": tvec.flatten(),
        "obj_pts": obj_pts,
        "img_pts": img_pts,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "rvec": rvec,
        "tvec": tvec,
        "n_corners": int(len(charuco_ids)),
    }


def _save_debug_image(path, rgb, detection, K, dist, board, square_length_m):
    """Write an annotated debug image. detection may be None (failed)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if detection is None:
        cv2.putText(bgr, "NO BOARD DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        if detection["marker_ids"] is not None and len(detection["marker_ids"]) > 0:
            cv2.aruco.drawDetectedMarkers(bgr, detection["marker_corners"], detection["marker_ids"])
        cv2.aruco.drawDetectedCornersCharuco(
            bgr, detection["charuco_corners"], detection["charuco_ids"], cornerColor=(0, 255, 0)
        )
        cv2.drawFrameAxes(bgr, K, dist, detection["rvec"], detection["tvec"], square_length_m * 2)
        cv2.putText(bgr, f"corners: {detection['n_corners']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(path), bgr)


def _annotate_preview(rgb, detection, K, dist, axis_length_m, num_captured):
    """Return a BGR preview image with detection overlay + status."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if detection is not None:
        if detection["marker_ids"] is not None and len(detection["marker_ids"]) > 0:
            cv2.aruco.drawDetectedMarkers(bgr, detection["marker_corners"], detection["marker_ids"])
        cv2.aruco.drawDetectedCornersCharuco(
            bgr, detection["charuco_corners"], detection["charuco_ids"], cornerColor=(0, 255, 0)
        )
        cv2.drawFrameAxes(bgr, K, dist, detection["rvec"], detection["tvec"], axis_length_m)
        status = f"corners: {detection['n_corners']}  [READY]"
        color = (0, 255, 0)
    else:
        status = "NO BOARD"
        color = (0, 0, 255)
    cv2.putText(bgr, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(bgr, f"captured: {num_captured}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bgr, "SPACE=capture  ENTER=done  Q=abort", (20, bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bgr


# ---------------------------------------------------------------------------
# Calibration validation
# ---------------------------------------------------------------------------


def _compute_quality(samples, T_cam2base, T_board2gripper, K, dist):
    """Return dict of error statistics."""
    R_cam2base = T_cam2base[:3, :3]
    t_cam2base = T_cam2base[:3, 3]
    R_board2gripper = T_board2gripper[:3, :3]
    t_board2gripper = T_board2gripper[:3, 3]

    # Per-pose board origin in base, recovered via camera chain.
    board_origins_via_cam = []
    reproj_errors_px = []
    for s in samples:
        R_g2b, t_g2b = s["R_gripper2base"], s["t_gripper2base"]
        R_t2c, t_t2c = s["R_target2cam"], s["t_target2cam"]

        # board origin in base via camera
        t_b_via_cam = R_cam2base @ t_t2c + t_cam2base
        board_origins_via_cam.append(t_b_via_cam)

        # Expected board pose in cam given current calibration:
        # board → gripper → base → cam
        R_b2cam_expected = R_cam2base.T @ R_g2b @ R_board2gripper
        t_b2cam_expected = R_cam2base.T @ (R_g2b @ t_board2gripper + t_g2b - t_cam2base)

        # Reproject the detected ChArUco corners using the expected pose.
        obj = s["obj_pts"]   # board-frame 3D corners
        img = s["img_pts"]   # detected 2D pixel positions
        rvec, _ = cv2.Rodrigues(R_b2cam_expected)
        proj, _ = cv2.projectPoints(obj, rvec, t_b2cam_expected, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        reproj_errors_px.extend(err.tolist())

    board_origins_via_cam = np.array(board_origins_via_cam)
    board_position_std_m = board_origins_via_cam.std(axis=0)

    return {
        "reprojection_error_mean_px": float(np.mean(reproj_errors_px)),
        "reprojection_error_max_px": float(np.max(reproj_errors_px)),
        "board_position_std_mm": (board_position_std_m * 1000.0).tolist(),
    }


def _compute_eye_in_hand_quality(samples, T_cam2gripper, T_board2base, K, dist):
    """Return quality stats for a moving camera and fixed board."""
    board_origins_via_cam = []
    reproj_errors_px = []

    for s in samples:
        R_g2b, t_g2b = s["R_gripper2base"], s["t_gripper2base"]
        R_t2c, t_t2c = s["R_target2cam"], s["t_target2cam"]

        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_g2b
        T_gripper2base[:3, 3] = t_g2b

        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_t2c
        T_target2cam[:3, 3] = t_t2c
        board_origins_via_cam.append((T_gripper2base @ T_cam2gripper @ T_target2cam)[:3, 3])

        T_cam2base = T_gripper2base @ T_cam2gripper
        T_target2cam_expected = _invert_transform(T_cam2base) @ T_board2base

        obj = s["obj_pts"]
        img = s["img_pts"]
        rvec, _ = cv2.Rodrigues(T_target2cam_expected[:3, :3])
        proj, _ = cv2.projectPoints(obj, rvec, T_target2cam_expected[:3, 3], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        reproj_errors_px.extend(err.tolist())

    board_origins_via_cam = np.array(board_origins_via_cam)
    board_position_std_m = board_origins_via_cam.std(axis=0)

    return {
        "reprojection_error_mean_px": float(np.mean(reproj_errors_px)),
        "reprojection_error_max_px": float(np.max(reproj_errors_px)),
        "board_position_std_mm": (board_position_std_m * 1000.0).tolist(),
    }


def _per_frame_mean_reproj_errors(samples, T_cam2base, T_board2gripper, K, dist):
    """Per-frame mean reprojection error (px) for eye-on-base setup."""
    R_cam2base = T_cam2base[:3, :3]
    t_cam2base = T_cam2base[:3, 3]
    R_board2gripper = T_board2gripper[:3, :3]
    t_board2gripper = T_board2gripper[:3, 3]
    errors = []
    for s in samples:
        R_g2b, t_g2b = s["R_gripper2base"], s["t_gripper2base"]
        R_b2cam = R_cam2base.T @ R_g2b @ R_board2gripper
        t_b2cam = R_cam2base.T @ (R_g2b @ t_board2gripper + t_g2b - t_cam2base)
        rvec, _ = cv2.Rodrigues(R_b2cam)
        proj, _ = cv2.projectPoints(s["obj_pts"], rvec, t_b2cam, K, dist)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - s["img_pts"].reshape(-1, 2), axis=1).mean())
    return np.array(errors)


def _per_frame_mean_reproj_errors_eye_in_hand(samples, T_cam2gripper, T_board2base, K, dist):
    """Per-frame mean reprojection error (px) for eye-in-hand setup."""
    errors = []
    for s in samples:
        T_gripper2base = _make_transform(s["R_gripper2base"], s["t_gripper2base"])
        T_target2cam_expected = _invert_transform(T_gripper2base @ T_cam2gripper) @ T_board2base
        rvec, _ = cv2.Rodrigues(T_target2cam_expected[:3, :3])
        proj, _ = cv2.projectPoints(s["obj_pts"], rvec, T_target2cam_expected[:3, 3], K, dist)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - s["img_pts"].reshape(-1, 2), axis=1).mean())
    return np.array(errors)


def _iqr_inlier_mask(errors, k=1.5):
    """Tukey IQR fence: True = inlier (error <= Q3 + k*IQR)."""
    q1, q3 = np.percentile(errors, [25, 75])
    return errors <= q3 + k * (q3 - q1)


# ---------------------------------------------------------------------------
# Capture: kinesthetic manual
# ---------------------------------------------------------------------------


def _run_calibration_pointcloud_viewer(
    cfg: DictConfig,
    camera_mount: str,
    extrinsics_path: Path,
    robot,
) -> None:
    vc = cfg.get("visualization", {}).get("viser", {})
    pc = vc.get("pointclouds", {}).get(camera_mount, {})
    if not pc:
        print(f"  [viser] No visualization.viser.pointclouds.{camera_mount} config found.")
        return

    camera = make_zed_camera(cfg, camera_mount)
    try:
        camera.run()

        from clear_franka.visualization import CortadoViserVisualizer

        visualizer = CortadoViserVisualizer(
            host=vc.get("host", "0.0.0.0"),
            port=int(vc.get("port", 8080)),
        )
        try:
            state = robot.get_last_teleop_state()
            visualizer.update(np.asarray(state["q"], dtype=float))
        except Exception as exc:
            print(f"  [viser] Could not update robot state: {exc}")

        if camera_mount == "hand":
            frame_name = pc.get("frame_name", "hand_zed")
            visualizer.add_hand_camera_frame_from_extrinsics(frame_name, extrinsics_path)
        else:
            frame_name = pc.get("frame_name", "/third_person_zed")
            visualizer.add_camera_frame_from_extrinsics(frame_name, extrinsics_path)

        camera.start_pointcloud_stream(
            update_hz=float(pc.get("update_hz", 5.0)),
            stride=int(pc.get("stride", 4)),
            max_points=int(pc.get("max_points", 100_000)),
            max_distance_m=float(pc.get("max_distance_m", 3.0)),
        )
        print(f"  [viser] {camera_mount} point cloud viewer running from {extrinsics_path}.")
        print("  [viser] Press ENTER to stop.")

        last_pointcloud_timestamp = None
        while True:
            latest_pointcloud = camera.get_latest_pointcloud()
            if latest_pointcloud is not None:
                points, colors, timestamp = latest_pointcloud
                if timestamp != last_pointcloud_timestamp:
                    visualizer.update_pointcloud(
                        frame_name,
                        points,
                        colors,
                        point_size=float(pc.get("point_size", 0.01)),
                    )
                    last_pointcloud_timestamp = timestamp
            if wait_for_enter(0.05):
                break
            time.sleep(0.02)
    finally:
        camera.stop_pointcloud_stream()
        camera.close()


def _capture_kinesthetic(robot, camera, cal, detector, board, K, dist, debug_dir):
    """Low-impedance kinesthetic teaching with live camera preview.

    The arm becomes back-drivable via a low-stiffness joint impedance
    controller running in a local background thread. The main thread
    runs the camera preview at the camera's native rate. The operator
    physically moves the arm to each desired calibration pose and presses:

        SPACE — capture current frame + EE pose as a sample
        ENTER — finish capture, proceed to hand-eye solve
        Q     — abort
    """
    kin = cal.kinesthetic

    session = robot.start_joint_impedance_session(
        hold_current_joint,
        period=0.001,
        stiffness=[float(v) for v in kin.joint_stiffness],
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
    )

    print()
    print("=" * 70)
    print("KINESTHETIC CAPTURE — the arm is now compliant.")
    if cal.get("camera_mount", "third_person") == "hand":
        print("Fix the ChArUco board RIGIDLY in the robot base/world.")
        print("Keep the hand camera pointed at the board while varying wrist pose.")
    else:
        print("Mount the ChArUco board RIGIDLY to the gripper.")
    print("Physically guide the arm to each calibration pose, then press:")
    print("  SPACE  → capture current frame + EE pose")
    print("  ENTER  → finish (need >= 4 captures to solve)")
    print("  Q      → abort without solving")
    print("=" * 70)
    print()

    samples = []
    frame_count = 0
    try:
        while True:
            frame_count += 1
            if frame_count % 30 == 0:
                status = session.status()
                if not status.get("running", True):
                    raise RuntimeError(f"Impedance tracker faulted: {status.get('error')}")

            frame = camera.grab_frame()
            if frame is None:
                # tiny yield to keep the GUI responsive
                cv2.waitKey(1)
                continue
            rgb, _ = frame

            detection = _detect_board_pose(detector, board, rgb, K, dist, cal.min_charuco_corners)
            preview = _annotate_preview(
                rgb, detection, K, dist,
                axis_length_m=cal.board.square_length_m * 2.0,
                num_captured=len(samples),
            )
            cv2.imshow("Calibration preview", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if detection is None:
                    logger.warning("Capture rejected: board not detected.")
                    continue
                state = robot.get_last_teleop_state()
                O_T_EE = np.asarray(state["O_T_EE"], dtype=float)
                R_gripper2base = O_T_EE[:3, :3].copy()
                t_gripper2base = O_T_EE[:3, 3].copy()
                samples.append({
                    "R_gripper2base": R_gripper2base,
                    "t_gripper2base": t_gripper2base,
                    **detection,
                })
                logger.info("  captured sample %d (%d corners)", len(samples), detection["n_corners"])
                if debug_dir is not None:
                    _save_debug_image(
                        debug_dir / f"manual_{len(samples):02d}_ok.png",
                        rgb, detection, K, dist, board, cal.board.square_length_m,
                    )
            elif key in (13, 10):  # Enter
                break
            elif key == ord('q'):
                samples = []
                break
    finally:
        cv2.destroyAllWindows()
        session.stop()

    return samples


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_calibration(cfg: DictConfig):
    cal = cfg.calibration
    camera_mount = cal.get("camera_mount", "third_person")
    if camera_mount not in ("third_person", "hand"):
        raise ValueError(
            f"calibration.camera_mount must be one of third_person|hand (got {camera_mount!r})"
        )

    out_path = Path(cfg.get("data_dir", "./data")) / f"extrinsics_{camera_mount}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(cfg.get("data_dir", "./data")) / f"calibration_debug_{camera_mount}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ camera
    camera_cfg = get_camera_config(cfg, camera_mount)
    camera = ZedCamera(
        resolution="HD2K",
        fps=camera_cfg["fps"],
        depth_mode=camera_cfg["depth_mode"],
        serial_number=camera_cfg["serial_number"],
        camera_id=camera_cfg["name"],
    )
    K, dist = camera.get_intrinsics()
    logger.info(
        "Opened %s ZED serial=%s; intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
        camera_mount,
        camera_cfg["serial_number"],
        K[0, 0],
        K[1, 1],
        K[0, 2],
        K[1, 2],
    )

    board, detector = _build_board(cal.board)

    # ------------------------------------------------------------------- robot
    logger.info("Connecting to Franka at %s ...", cfg.robot.ip)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    # ------------------------------------------------------------------- capture
    try:
        samples = _capture_kinesthetic(robot, camera, cal, detector, board, K, dist, debug_dir)
    finally:
        camera.close()

    if len(samples) < 4:
        raise RuntimeError(
            f"Only {len(samples)} usable samples captured; need >= 4 (more is better). "
            "Check board visibility and lighting."
        )

    # ---------------------------------------------------------------- solve
    n_captured = len(samples)
    logger.info("Running hand-eye calibration over %d samples ...", n_captured)
    R_g2b = [s["R_gripper2base"] for s in samples]
    t_g2b = [s["t_gripper2base"] for s in samples]
    R_t2c = [s["R_target2cam"] for s in samples]
    t_t2c = [s["t_target2cam"] for s in samples]

    payload = {
        "camera_mount": camera_mount,
        "camera_id": camera_cfg["name"],
        "camera_serial_number": camera_cfg["serial_number"],
        "intrinsics": {"K": K.tolist(), "dist": dist.tolist()},
        "image_size": [int(camera._img_w), int(camera._img_h)],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_poses_captured": n_captured,
        "board": {
            "squares_x": cal.board.squares_x,
            "squares_y": cal.board.squares_y,
            "square_length_m": cal.board.square_length_m,
            "marker_length_m": cal.board.marker_length_m,
            "dictionary": cal.board.dictionary,
        },
    }

    # Initial solve over all samples to obtain per-frame errors for outlier rejection.
    if camera_mount == "hand":
        T_cam2gripper = _solve_eye_in_hand(R_g2b, t_g2b, R_t2c, t_t2c)
        T_gripper2cam = _invert_transform(T_cam2gripper)
        T_board2base, _ = _solve_board2base(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper)
        per_frame_errors = _per_frame_mean_reproj_errors_eye_in_hand(
            samples, T_cam2gripper, T_board2base, K, dist
        )
    else:
        T_cam2base = _solve_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
        T_board2gripper, _ = _solve_board2gripper(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2base)
        per_frame_errors = _per_frame_mean_reproj_errors(
            samples, T_cam2base, T_board2gripper, K, dist
        )

    # IQR-based outlier rejection: re-solve if any frames are removed.
    inlier_mask = _iqr_inlier_mask(per_frame_errors)
    n_outliers = int((~inlier_mask).sum())
    if n_outliers > 0:
        logger.info("Outlier rejection: removing %d/%d samples (IQR k=1.5 fence):", n_outliers, n_captured)
        for i, (err, is_inlier) in enumerate(zip(per_frame_errors, inlier_mask)):
            logger.info("  sample %02d: %.2f px%s", i + 1, err, "" if is_inlier else "  <-- outlier")
        samples = [s for s, m in zip(samples, inlier_mask) if m]
        if len(samples) < 4:
            raise RuntimeError(
                f"Only {len(samples)} samples remain after outlier rejection; need >= 4. "
                "Recapture with better board visibility."
            )
        R_g2b = [s["R_gripper2base"] for s in samples]
        t_g2b = [s["t_gripper2base"] for s in samples]
        R_t2c = [s["R_target2cam"] for s in samples]
        t_t2c = [s["t_target2cam"] for s in samples]
        if camera_mount == "hand":
            T_cam2gripper = _solve_eye_in_hand(R_g2b, t_g2b, R_t2c, t_t2c)
            T_gripper2cam = _invert_transform(T_cam2gripper)
            T_board2base, _ = _solve_board2base(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper)
        else:
            T_cam2base = _solve_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
            T_board2gripper, _ = _solve_board2gripper(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2base)

    payload["num_poses_used"] = len(samples)

    if camera_mount == "hand":
        quality = _compute_eye_in_hand_quality(samples, T_cam2gripper, T_board2base, K, dist)
        logger.info("T_gripper2cam translation: %s m", T_gripper2cam[:3, 3].tolist())
        logger.info("T_gripper2cam rotation (xyz Euler): %s rad",
                    R.from_matrix(T_gripper2cam[:3, :3]).as_euler("xyz").tolist())
        payload.update({
            "T_gripper2cam": T_gripper2cam.tolist(),
            "T_cam2gripper": T_cam2gripper.tolist(),
            "T_board2base": T_board2base.tolist(),
        })
    else:
        quality = _compute_quality(samples, T_cam2base, T_board2gripper, K, dist)
        logger.info("T_cam2base translation: %s m", T_cam2base[:3, 3].tolist())
        logger.info("T_cam2base rotation (xyz Euler): %s rad",
                    R.from_matrix(T_cam2base[:3, :3]).as_euler("xyz").tolist())
        payload.update({
            "T_cam2base": T_cam2base.tolist(),
            "T_board2gripper": T_board2gripper.tolist(),
        })

    payload.update(quality)
    logger.info("Mean reproj err: %.3f px (max %.3f)",
                quality["reprojection_error_mean_px"], quality["reprojection_error_max_px"])
    logger.info("Board position std: %s mm", quality["board_position_std_mm"])

    # ---------------------------------------------------------------- gate
    ok = True
    if quality["reprojection_error_mean_px"] > cal.reprojection_error_max_px:
        logger.warning("Mean reprojection error exceeds threshold (%.2f > %.2f px)",
                       quality["reprojection_error_mean_px"], cal.reprojection_error_max_px)
        ok = False
    if any(s > cal.board_position_std_max_mm for s in quality["board_position_std_mm"]):
        logger.warning("Board position std exceeds threshold (%s > %.2f mm)",
                       quality["board_position_std_mm"], cal.board_position_std_max_mm)
        ok = False
    if not ok:
        logger.warning("Calibration quality below thresholds.")
        try:
            response = input("Save calibration anyway? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = ""
            print()
        if response not in {"y", "yes"}:
            logger.warning("Discarding calibration; no file written.")
            return

    # ---------------------------------------------------------------- save
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_path)

    try:
        show_pointcloud = prompt_yes_no(
            "View camera point cloud in viser with extrinsics applied?",
            default=False,
        )
    except (EOFError, KeyboardInterrupt):
        show_pointcloud = False
        print()

    if show_pointcloud:
        try:
            _run_calibration_pointcloud_viewer(cfg, camera_mount, out_path, robot)
        except KeyboardInterrupt:
            print()
        except Exception as e:
            print(f"  [viser] Point cloud viewer failed: {e}")
