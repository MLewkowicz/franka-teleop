"""Third-person ZED 2 → robot base extrinsics calibration.

Eye-on-base hand-eye calibration: the ChArUco board is rigidly attached
to the gripper at an unknown offset; the third-person ZED 2 is stationary.
At each captured pose we record the end-effector pose and a synchronized
camera frame, detect the ChArUco board, then solve ``cv2.calibrateHandEye``
for ``T_cam2base``.

Three capture modes (set via ``calibration.mode`` in config):
  * ``manual``  — low Cartesian impedance + live camera preview. The
                  operator physically guides the arm to each pose and
                  presses SPACE to capture. Captured EE poses are saved
                  to ``calibration.waypoints_file`` for later replay.
  * ``replay``  — load waypoints from ``waypoints_file`` and drive the
                  robot through them automatically.
  * ``sweep``   — generate waypoints from the home/sweep config block
                  and drive automatically (no saved file needed).

Output: ``data/extrinsics_third_person.json`` with the 4x4 transform,
the recovered board-on-gripper offset, intrinsics snapshot, and quality
statistics.
"""

import datetime
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from net_franky.franky import Affine, CartesianImpedanceTracker, CartesianMotion, Robot

from camera import ZedCamera

logger = logging.getLogger(__name__)

DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


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
# Waypoint file I/O
# ---------------------------------------------------------------------------


def _save_waypoints_file(path: Path, samples: list, source_mode: str):
    """Persist captured EE poses to JSON for later replay.

    ``samples`` is the in-memory sample list (each has R_gripper2base, t_gripper2base).
    Stored as position + quat_xyzw so they can be re-fed to franky's Affine.
    """
    waypoints = []
    for s in samples:
        R_g2b = np.asarray(s["R_gripper2base"])
        t_g2b = np.asarray(s["t_gripper2base"])
        waypoints.append({
            "position": t_g2b.tolist(),
            "quat_xyzw": R.from_matrix(R_g2b).as_quat().tolist(),
        })
    payload = {
        "version": 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_mode": source_mode,
        "waypoints": waypoints,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %d waypoints to %s", len(waypoints), path)


def _load_waypoints_file(path: Path):
    """Return a list of (position np.array[3], quat_xyzw np.array[4])."""
    with path.open("r") as f:
        payload = json.load(f)
    out = []
    for w in payload["waypoints"]:
        out.append((np.asarray(w["position"], dtype=float), np.asarray(w["quat_xyzw"], dtype=float)))
    return out


# ---------------------------------------------------------------------------
# Waypoint generation (sweep mode)
# ---------------------------------------------------------------------------


def _build_waypoints(cfg: DictConfig):
    """Compose home pose with paired position/rotation perturbations.

    Returns a list of (position [3], quaternion xyzw [4]) tuples.
    """
    home_pos = np.array(cfg.home.position, dtype=float)
    R_home = R.from_euler("xyz", np.array(cfg.home.euler_xyz, dtype=float))

    pos_offsets = [np.array(p, dtype=float) for p in cfg.sweep.position_offsets_m]
    rot_offsets = [R.from_euler("xyz", np.array(r, dtype=float)) for r in cfg.sweep.rotation_offsets_rad]

    n = min(len(pos_offsets), len(rot_offsets))
    if len(pos_offsets) != len(rot_offsets):
        logger.warning(
            "sweep position_offsets (%d) and rotation_offsets (%d) differ in length; "
            "using first %d of each.",
            len(pos_offsets), len(rot_offsets), n,
        )

    waypoints = []
    for i in range(n):
        position = home_pos + pos_offsets[i]
        R_target = R_home * rot_offsets[i]  # local-frame composition
        waypoints.append((position, R_target.as_quat()))
    return waypoints


# ---------------------------------------------------------------------------
# Hand-eye solve & validation
# ---------------------------------------------------------------------------


def _solve_hand_eye(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
    """Eye-on-base configuration: pass base2gripper to recover cam2base."""
    R_base2gripper = [R_g2b.T for R_g2b in R_gripper2base_list]
    t_base2gripper = [-R_g2b.T @ t_g2b for R_g2b, t_g2b in zip(R_gripper2base_list, t_gripper2base_list)]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_base2gripper,
        t_gripper2base=t_base2gripper,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()
    return T


def _solve_board2gripper(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list, T_cam2base):
    """Recover the (constant) board-on-gripper transform by averaging per-pose estimates."""
    R_cam2base = T_cam2base[:3, :3]
    t_cam2base = T_cam2base[:3, 3]

    Ts = []
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(
        R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list
    ):
        # board → base via camera chain
        R_t2b = R_cam2base @ R_t2c
        t_t2b = R_cam2base @ t_t2c + t_cam2base
        # base → gripper
        R_b2g = R_g2b.T
        t_b2g = -R_g2b.T @ t_g2b
        # board → gripper
        R_t2g = R_b2g @ R_t2b
        t_t2g = R_b2g @ t_t2b + t_b2g
        T = np.eye(4)
        T[:3, :3] = R_t2g
        T[:3, 3] = t_t2g
        Ts.append(T)

    # Average rotations via quaternions, translations via arithmetic mean.
    quats = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    # Flip signs so all quats are in the same hemisphere as the first
    flip = np.sign(quats @ quats[0])
    flip[flip == 0] = 1
    quats = quats * flip[:, None]
    mean_quat = quats.mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    T_board2gripper = np.eye(4)
    T_board2gripper[:3, :3] = R.from_quat(mean_quat).as_matrix()
    T_board2gripper[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return T_board2gripper, Ts


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


# ---------------------------------------------------------------------------
# Capture: automated (replay / sweep) and kinesthetic (manual)
# ---------------------------------------------------------------------------


def _capture_automated(robot, camera, waypoints, cal, detector, board, K, dist, debug_dir):
    """Drive the robot through each waypoint and capture (pose, image) samples."""
    samples = []
    for i, (pos, quat) in enumerate(waypoints):
        logger.info("[%2d/%d] Moving to position=%s ...", i + 1, len(waypoints), pos.tolist())
        try:
            robot.move(CartesianMotion(Affine(pos.tolist(), quat.tolist())))
        except Exception as e:
            logger.warning("  motion failed: %s — skipping waypoint.", e)
            continue

        time.sleep(cal.settle_seconds)

        frame = camera.grab_frame()
        if frame is None:
            logger.warning("  camera grab failed — skipping waypoint.")
            continue
        rgb, _ = frame

        ee_pose = robot.current_cartesian_state.pose.end_effector_pose
        R_gripper2base = np.array(ee_pose.matrix)[:3, :3]
        t_gripper2base = np.array(ee_pose.translation)

        detection = _detect_board_pose(detector, board, rgb, K, dist, cal.min_charuco_corners)

        if debug_dir is not None:
            tag = "ok" if detection is not None else "fail"
            _save_debug_image(
                debug_dir / f"waypoint_{i:02d}_{tag}.png",
                rgb, detection, K, dist, board, cal.board.square_length_m,
            )

        if detection is None:
            logger.warning("  board not detected (need >=%d corners) — skipping waypoint.", cal.min_charuco_corners)
            continue

        samples.append({
            "R_gripper2base": R_gripper2base,
            "t_gripper2base": t_gripper2base,
            **detection,
        })
        logger.info("  captured (%d charuco corners detected).", detection["n_corners"])
    return samples


def _capture_kinesthetic(robot, camera, cal, detector, board, K, dist, debug_dir):
    """Low-impedance kinesthetic teaching with live camera preview.

    The arm becomes back-drivable via a near-zero-stiffness Cartesian
    impedance controller running in a background thread. The main thread
    runs the camera preview at the camera's native rate. The operator
    physically moves the arm to each desired calibration pose and presses:

        SPACE — capture current frame + EE pose as a sample
        ENTER — finish capture, proceed to hand-eye solve
        Q     — abort
    """
    kin = cal.kinesthetic
    stop_event = threading.Event()
    tracker_error = [None]

    def _hold_compliant():
        try:
            with CartesianImpedanceTracker(
                robot,
                translational_stiffness=kin.translational_stiffness,
                rotational_stiffness=kin.rotational_stiffness,
                nullspace_stiffness=kin.nullspace_stiffness,
                lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
                upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
                period=kin.period,
            ) as tracker:
                while not stop_event.is_set() and tracker.tick():
                    # Keep target glued to current pose → no restoring force.
                    tracker.set_target(tracker.current_pose.end_effector_pose)
        except Exception as e:  # noqa: BLE001
            tracker_error[0] = e

    thread = threading.Thread(target=_hold_compliant, daemon=True)
    thread.start()

    # Give the tracker a beat to come online.
    time.sleep(0.2)
    if tracker_error[0] is not None:
        raise RuntimeError(f"Impedance tracker failed to start: {tracker_error[0]}")

    print()
    print("=" * 70)
    print("KINESTHETIC CAPTURE — the arm is now compliant.")
    print("Physically guide the arm to each calibration pose, then press:")
    print("  SPACE  → capture current frame + EE pose")
    print("  ENTER  → finish (need >= 4 captures to solve)")
    print("  Q      → abort without solving")
    print("=" * 70)
    print()

    samples = []
    try:
        while True:
            if tracker_error[0] is not None:
                raise RuntimeError(f"Impedance tracker faulted: {tracker_error[0]}")

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
                ee_pose = robot.current_cartesian_state.pose.end_effector_pose
                R_gripper2base = np.array(ee_pose.matrix)[:3, :3]
                t_gripper2base = np.array(ee_pose.translation)
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
        stop_event.set()
        thread.join(timeout=2.0)

    return samples


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_calibration(cfg: DictConfig):
    cal = cfg.calibration
    out_path = Path(cal.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(cal.debug_dir) if cal.get("debug_dir") else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    mode = cal.mode
    if mode not in ("manual", "replay", "sweep"):
        raise ValueError(f"calibration.mode must be one of manual|replay|sweep (got {mode!r})")

    # ------------------------------------------------------------------ camera
    logger.info("Opening third-person ZED 2 ...")
    camera = ZedCamera(
        resolution=cfg.camera.resolution,
        fps=cfg.camera.fps,
        depth_mode=cfg.camera.depth_mode,
    )
    K, dist = camera.get_intrinsics()
    logger.info("ZED intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f", K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    board, detector = _build_board(cal.board)

    # ------------------------------------------------------------------- robot
    logger.info("Connecting to Franka at %s ...", cfg.robot.ip)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = cal.relative_dynamics_factor

    # ------------------------------------------------------------------- capture
    try:
        if mode == "manual":
            samples = _capture_kinesthetic(robot, camera, cal, detector, board, K, dist, debug_dir)
        else:
            if mode == "replay":
                wp_path = Path(cal.waypoints_file)
                if not wp_path.exists():
                    raise FileNotFoundError(
                        f"replay mode requires {wp_path} — record waypoints with mode=manual first."
                    )
                waypoints = _load_waypoints_file(wp_path)
                logger.info("Loaded %d waypoints from %s", len(waypoints), wp_path)
            else:  # sweep
                waypoints = _build_waypoints(cal)
                logger.info("%d sweep waypoints generated.", len(waypoints))

            print()
            print("=" * 70)
            print("Mount the ChArUco board RIGIDLY to the gripper.")
            print("The robot will move SLOWLY through %d waypoints." % len(waypoints))
            print("Press ENTER to begin (Ctrl-C to abort).")
            print("=" * 70)
            input()

            samples = _capture_automated(
                robot, camera, waypoints, cal, detector, board, K, dist, debug_dir,
            )
    finally:
        camera.close()

    if len(samples) < 4:
        raise RuntimeError(
            f"Only {len(samples)} usable waypoints captured; need >= 4 (more is better). "
            "Check board visibility and lighting."
        )

    # Persist the captured EE poses for replay-mode reruns.
    if mode == "manual" and cal.get("waypoints_file"):
        _save_waypoints_file(Path(cal.waypoints_file), samples, source_mode="manual")

    # ---------------------------------------------------------------- solve
    logger.info("Running hand-eye calibration over %d samples ...", len(samples))
    R_g2b = [s["R_gripper2base"] for s in samples]
    t_g2b = [s["t_gripper2base"] for s in samples]
    R_t2c = [s["R_target2cam"] for s in samples]
    t_t2c = [s["t_target2cam"] for s in samples]

    T_cam2base = _solve_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
    T_board2gripper, _ = _solve_board2gripper(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2base)
    quality = _compute_quality(samples, T_cam2base, T_board2gripper, K, dist)

    logger.info("T_cam2base translation: %s m", T_cam2base[:3, 3].tolist())
    logger.info("T_cam2base rotation (xyz Euler): %s rad",
                R.from_matrix(T_cam2base[:3, :3]).as_euler("xyz").tolist())
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
        logger.warning("Calibration quality below thresholds. Saving anyway as '*.bad.json'.")
        out_path = out_path.with_suffix(".bad.json")

    # ---------------------------------------------------------------- save
    payload = {
        "camera_id": "third_person_zed2_idx0",
        "T_cam2base": T_cam2base.tolist(),
        "T_board2gripper": T_board2gripper.tolist(),
        "intrinsics": {"K": K.tolist(), "dist": dist.tolist()},
        "image_size": [int(camera._img_w), int(camera._img_h)],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_poses_used": len(samples),
        **quality,
        "board": {
            "squares_x": cal.board.squares_x,
            "squares_y": cal.board.squares_y,
            "square_length_m": cal.board.square_length_m,
            "marker_length_m": cal.board.marker_length_m,
            "dictionary": cal.board.dictionary,
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_path)
