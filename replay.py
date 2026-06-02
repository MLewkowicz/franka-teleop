"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from clear_franka.episode_io import find_latest_episode, load_episode

from zero_franky import Robot
from franky import Affine, JointMotion, JointState, ManipulabilityTask, PostureTask, Twist
from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    joint_friction_kwargs,
)
from clear_franka.cartesian_trajectory import CartesianTrajectory
from clear_franka.geometry import pack_Rp
from clear_franka.ik import CartesianIK, ik_frame_from_episode
from clear_franka.joint_trajectory import Trajectory
from clear_franka.preprocess import preprocess_episode_arrays
from clear_franka.utils import resolve_project_path


def prompt_reverse_reset() -> bool:
    try:
        answer = input("  Play trajectory in reverse to reset robot? [Y/n] ").strip().lower()
    except EOFError:
        print("  Reverse reset skipped (no input available).")
        return False
    return answer in ("", "y", "yes")


def _preprocess_kwargs(pre_cfg: DictConfig) -> dict:
    params = OmegaConf.to_container(pre_cfg, resolve=True)
    trim_cfg = params.get("trim", {})
    retime_cfg = params.get("retime", {})
    smooth_cfg = params.get("smooth", {})
    return {
        "trim_enabled": bool(trim_cfg.get("enabled", True)),
        "trim_time_window": float(trim_cfg.get("time_window", 0.3)),
        "trim_threshold": float(trim_cfg.get("threshold", 0.01)),
        "retime_enabled": bool(retime_cfg.get("enabled", False)),
        "retime_sample_uniform": bool(retime_cfg.get("sample_uniform", False)),
        "retime_path_tol": retime_cfg.get("path_tol", None),
        "retime_max_joint_vel": retime_cfg.get("max_joint_vel", None),
        "retime_max_joint_accel": retime_cfg.get("max_joint_accel", None),
        "smooth_enabled": bool(smooth_cfg.get("enabled", True)),
        "smooth_max_joint_vel": smooth_cfg["max_joint_vel"],
        "smooth_max_joint_accel": smooth_cfg["max_joint_accel"],
        "smooth_max_joint_jerk": smooth_cfg["max_joint_jerk"],
        "smooth_dt": float(smooth_cfg.get("dt", 0.001)),
        "gripper_dwell_s": float(params.get("gripper_dwell_s", 0.0)),
    }


def play_joint_trajectory(
    *,
    robot: Robot,
    rc: DictConfig,
    gc,
    stiffness: np.ndarray,
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    has_joint_vel: bool,
    gripper,
    gripper_open_data,
    recorder=None,
    gripper_open_for_record=None,
    viz=None,
):
    n_steps = len(timestamps)
    trajectory = Trajectory(joint_pos, timestamps)

    with robot.start_joint_impedance_tracker(
        period=rc.period,
        stiffness=stiffness,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        **joint_friction_kwargs(rc, speed=float(rc.speed)),
    ) as session:
        step = 0
        replay_start = None
        last_gripper_open = None

        while True:
            if session.tick() is None:
                raise RuntimeError("Impedance tracker stopped before replay completed.")
            if replay_start is None:
                replay_start = time.monotonic()

            elapsed = (time.monotonic() - replay_start) * rc.speed

            while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                step += 1

            if elapsed >= timestamps[-1]:
                print("  Replay complete.")
                session.set_target(joint_pos[-1])
                break

            q = trajectory.interpolate(elapsed).reshape(7)
            if has_joint_vel:
                dq = np.asarray(trajectory._spline.derivative()(elapsed), dtype=float).reshape(7)
                dq = dq * rc.speed
                session.set_target(q, dq=dq)
            else:
                session.set_target(q)

            if gripper is not None and np.isfinite(gripper_open_data[step]):
                current_gripper_open = bool(round(float(gripper_open_data[step])))
                if current_gripper_open != last_gripper_open:
                    target_width = (
                        gc.get("open_width_m", 0.085)
                        if current_gripper_open
                        else gc.get("close_width_m", 0.0)
                    )
                    try:
                        gripper.move_width(
                            target_width,
                            speed=int(gc.get("speed", 255)),
                            force=int(gc.get("force", 255)),
                            wait=False,
                            max_width_m=gc.get("max_width_m", 0.085),
                        )
                        last_gripper_open = current_gripper_open
                        gripper_open_for_record = current_gripper_open
                    except Exception as e:
                        print(f"\n  [gripper] move failed: {e}")

            if recorder is not None:
                teleop_state = session.state
                if teleop_state is None:
                    teleop_state = robot.wait_for_state(timeout=5.0)
                measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                recorder.step(
                    ee_pos=measured_pose[:3, 3],
                    ee_rot=measured_pose[:3, :3],
                    cmd_linear_vel=np.zeros(3),
                    cmd_angular_vel=np.zeros(3),
                    buttons=0,
                    enabled=True,
                    joint_pos=np.asarray(teleop_state["q"], dtype=float),
                    joint_vel=np.asarray(teleop_state["dq"], dtype=float),
                    gripper_open=gripper_open_for_record,
                    robot_abs_time=float(teleop_state["abs_time"]),
                )

            if viz is not None:
                viz.step()

    return gripper_open_for_record


def play_cartesian_trajectory(
    *,
    robot: Robot,
    cfg: DictConfig,
    rc: DictConfig,
    gc,
    timestamps: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    joint_pos: np.ndarray,
    gripper,
    gripper_open_data,
    recorder=None,
    gripper_open_for_record=None,
    viz=None,
):
    n_steps = len(timestamps)
    trajectory = CartesianTrajectory(ee_pos, ee_rot, timestamps)
    nullspace_target_cfg = rc.get("nullspace_target", None)
    if nullspace_target_cfg is None:
        nullspace_target = np.asarray(joint_pos[0], dtype=float)
    elif str(nullspace_target_cfg).lower() == "none":
        nullspace_target = None
    else:
        nullspace_target = np.asarray(nullspace_target_cfg, dtype=float)

    with robot.start_cartesian_impedance_tracker(
        period=rc.period,
        translational_stiffness=float(
            rc.get("translational_stiffness", cfg.teleop.translational_stiffness)
        ),
        rotational_stiffness=float(
            rc.get("rotational_stiffness", cfg.teleop.rotational_stiffness)
        ),
        posture_task=(
            PostureTask(
                nullspace_target,
                stiffness=float(
                    rc.get("nullspace_stiffness", cfg.teleop.nullspace_stiffness)
                ),
            )
            if nullspace_target is not None else None
        ),
        manipulability_task=ManipulabilityTask(gain=5.0, max_torque=1.0),
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
    ) as session:
        step = 0
        replay_start = None
        last_gripper_open = None

        while True:
            if session.tick() is None:
                raise RuntimeError("Impedance tracker stopped before replay completed.")
            if replay_start is None:
                replay_start = time.monotonic()

            elapsed = (time.monotonic() - replay_start) * rc.speed

            while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                step += 1

            if elapsed >= timestamps[-1]:
                print("  Replay complete.")
                session.set_target(Affine(pack_Rp(ee_rot[-1], ee_pos[-1])))
                break

            target_pos, target_rot = trajectory.interpolate(elapsed)
            linear_vel, angular_vel = trajectory.velocity(elapsed)
            session.set_target(
                Affine(pack_Rp(target_rot, target_pos)),
                Twist(linear_vel * rc.speed, angular_vel * rc.speed),
            )

            if gripper is not None and np.isfinite(gripper_open_data[step]):
                current_gripper_open = bool(round(float(gripper_open_data[step])))
                if current_gripper_open != last_gripper_open:
                    target_width = (
                        gc.get("open_width_m", 0.085)
                        if current_gripper_open
                        else gc.get("close_width_m", 0.0)
                    )
                    try:
                        gripper.move_width(
                            target_width,
                            speed=int(gc.get("speed", 255)),
                            force=int(gc.get("force", 255)),
                            wait=False,
                            max_width_m=gc.get("max_width_m", 0.085),
                        )
                        last_gripper_open = current_gripper_open
                        gripper_open_for_record = current_gripper_open
                    except Exception as e:
                        print(f"\n  [gripper] move failed: {e}")

            if recorder is not None:
                teleop_state = robot.get_last_teleop_state()
                measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                recorder.step(
                    ee_pos=measured_pose[:3, 3],
                    ee_rot=measured_pose[:3, :3],
                    cmd_linear_vel=np.zeros(3),
                    cmd_angular_vel=np.zeros(3),
                    buttons=0,
                    enabled=True,
                    joint_pos=np.asarray(teleop_state["q"], dtype=float),
                    joint_vel=np.asarray(teleop_state["dq"], dtype=float),
                    gripper_open=gripper_open_for_record,
                    robot_abs_time=float(teleop_state["abs_time"]),
                )

            if viz is not None:
                viz.step()

    return gripper_open_for_record


def play_with_recovery(*, play_fn=play_joint_trajectory, **kwargs):
    robot = kwargs["robot"]
    while True:
        robot.recover_from_errors()
        try:
            return play_fn(**kwargs)
        except RuntimeError as e:
            print(f"\n  Controller faulted: {e}")
            print("  Recovering and retrying...")


def _resolve_pointcloud_source(cfg: DictConfig):
    """Return (source_name, source_cfg, enabled) for the active point-cloud source."""
    vc = cfg.get("visualization", {})
    pointcloud_cfg = vc.get("pointclouds", {})
    source = next(
        (n for n in ("third_person", "hand") if pointcloud_cfg.get(n, {}).get("enabled", False)),
        "third_person",
    )
    pc = pointcloud_cfg.get(source, {})
    enabled = bool(vc.get("enabled", False) and pc.get("enabled", False))
    return source, pc, enabled


def _setup_replay_visualizer(cfg: DictConfig, cameras: dict, pointcloud_source: str, pc):
    """Build the viser visualizer + point-cloud stream when visualization.enabled.

    ``cameras`` is the already-running camera dict (may be empty). Returns
    (visualizer, pointcloud_camera, pointcloud_frame_name) or (None, None, None).
    """
    vc = cfg.get("visualization", {})
    if not vc.get("enabled", False):
        return None, None, None

    from clear_franka.visualization import CortadoViserVisualizer

    visualizer = CortadoViserVisualizer(host=vc.get("host", "0.0.0.0"), port=int(vc.get("port", 8080)))
    pointcloud_camera = cameras.get(pointcloud_source)
    pointcloud_frame_name = None
    if pointcloud_camera is not None and pc.get("enabled", False):
        frame_name = pc.get(
            "frame_name", "hand_zed" if pointcloud_source == "hand" else "/third_person_zed"
        )
        if pointcloud_source == "hand":
            pointcloud_frame_name = visualizer.add_hand_camera_frame_from_extrinsics(
                frame_name, cfg.cameras.hand.get("extrinsics_path", "./data/extrinsics_hand.json")
            )
        else:
            pointcloud_frame_name = visualizer.add_camera_frame_from_extrinsics(
                frame_name,
                cfg.cameras.third_person.get(
                    "extrinsics_path", "./data/extrinsics_third_person.json"
                ),
            )
        pointcloud_camera.start_pointcloud_stream(
            update_hz=float(pc.get("update_hz", 5.0)),
            stride=int(pc.get("stride", 4)),
            max_points=int(pc.get("max_points", 100_000)),
            max_distance_m=float(pc.get("max_distance_m", 3.0)),
        )
        print(f"  [viser] {pointcloud_source} ZED point cloud enabled.")
    return visualizer, pointcloud_camera, pointcloud_frame_name


class _ReplayVizUpdater:
    """Throttled per-tick viser update for the replay playback loops.

    Reads the latest controller state and pushes joints, the EE frame, and the
    point cloud to viser, rate-limited to ``update_hz`` so it doesn't burden the
    1 kHz reference-streaming loop.
    """

    def __init__(self, visualizer, robot, pointcloud_camera, pointcloud_frame_name, pc, update_hz):
        self._visualizer = visualizer
        self._robot = robot
        self._pointcloud_camera = pointcloud_camera
        self._pointcloud_frame_name = pointcloud_frame_name
        self._pc = pc or {}
        self._min_dt = (1.0 / float(update_hz)) if update_hz else 0.0
        self._last_update = 0.0
        self._last_pc_timestamp = None

    def step(self):
        now = time.monotonic()
        if now - self._last_update < self._min_dt:
            return
        self._last_update = now
        try:
            teleop_state = self._robot.get_last_teleop_state()
        except RuntimeError:
            return
        joint_pos = np.asarray(teleop_state["q"], dtype=float)
        measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
        self._visualizer.update(joint_pos)
        self._visualizer.update_eef_frame(measured_pose)
        if self._pointcloud_camera is not None:
            latest = self._pointcloud_camera.get_latest_pointcloud()
            if latest is not None:
                points, colors, timestamp = latest
                if timestamp != self._last_pc_timestamp:
                    self._visualizer.update_pointcloud(
                        self._pointcloud_frame_name or self._pc.get("frame_name", "/third_person_zed"),
                        points,
                        colors,
                        point_size=float(self._pc.get("point_size", 0.01)),
                    )
                    self._last_pc_timestamp = timestamp


def run_replay(cfg: DictConfig):
    rc = cfg.replay
    gc = cfg.get("gripper", {})

    if rc.episode is not None:
        episode_path = Path(rc.episode)
    else:
        episode_path = find_latest_episode(cfg.data_dir)

    print(f"Loading episode: {episode_path}")
    episode = load_episode(episode_path)
    viz_enabled = bool(cfg.get("visualization", {}).get("enabled", False))
    # EE position path at each preprocess stage, for the viser overlay below.
    ee_stages = None
    raw_ee_pos = episode.get("ee_pos")
    if bool(rc.get("preprocess", True)):
        pre_cfg = cfg.get("preprocess", None)
        if pre_cfg is None:
            raise RuntimeError("replay.preprocess=true requires the top-level preprocess config")
        print("  Preprocessing episode before replay...")
        processed = preprocess_episode_arrays(
            episode, collect_ee_stages=viz_enabled, **_preprocess_kwargs(pre_cfg)
        )
        if processed is None:
            raise RuntimeError("Replay preprocessing failed; see preprocess logs above")
        ee_stages = processed.pop("ee_stages", None)
        episode.update(processed)
    elif viz_enabled and raw_ee_pos is not None:
        # No preprocessing — only the original recorded path exists.
        ee_stages = {"original": np.asarray(raw_ee_pos, dtype=np.float64)[:, :3]}

    timestamps = episode["timestamps"]
    joint_pos = episode["joint_pos"]
    joint_vel = episode["joint_vel"]
    tracker_mode = str(rc.get("tracker", "joint")).lower()
    if tracker_mode not in {"joint", "cartesian", "ik"}:
        raise ValueError(f"replay.tracker must be 'joint', 'cartesian', or 'ik', got {tracker_mode!r}")
    n_steps = len(timestamps)
    duration = timestamps[-1]

    if tracker_mode != "ik" and np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )
    has_joint_vel = not np.any(np.isnan(joint_vel))
    ee_pos = episode.get("ee_pos")
    ee_rot = episode.get("ee_rot")
    if tracker_mode in {"cartesian", "ik"}:
        if ee_pos is None or ee_rot is None:
            raise ValueError(f"{tracker_mode} replay requires ee_pos and ee_rot datasets in the episode")
        if np.any(np.isnan(ee_pos)) or np.any(np.isnan(ee_rot)):
            raise ValueError(f"{tracker_mode} replay requires finite ee_pos and ee_rot samples")

    gripper_open_data = episode.get("gripper_open")
    has_gripper_data = (
        gripper_open_data is not None and not np.all(np.isnan(gripper_open_data))
    )

    stiffness = np.asarray(rc.joint_stiffness, dtype=float)
    if stiffness.shape != (7,):
        raise ValueError(f"replay.joint_stiffness must have 7 entries, got shape {stiffness.shape}")

    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Tracker: {tracker_mode}")
    if tracker_mode == "joint":
        print(f"  Joint stiffness: {stiffness.tolist()}")
        print(f"  Joint velocity feedforward: {'enabled' if has_joint_vel else 'disabled (NaN in recording)'}")
    elif tracker_mode == "ik":
        print(f"  Joint stiffness: {stiffness.tolist()}")
        print(f"  IK: Cartesian waypoints converted to a joint trajectory before playback")
    else:
        print(
            "  Cartesian stiffness: "
            f"translation={float(rc.get('translational_stiffness', cfg.teleop.translational_stiffness))}, "
            f"rotation={float(rc.get('rotational_stiffness', cfg.teleop.rotational_stiffness))}"
        )
    if has_gripper_data:
        print(f"  Gripper replay: {'enabled' if gc.get('enabled', False) else 'disabled (gripper.enabled=false in config)'}")
    else:
        print(f"  Gripper replay: disabled (no gripper data in episode)")
    print(f"  Recording: {'enabled' if rc.get('record', False) else 'disabled'}")
    print(f"  Press Ctrl-C to abort.")

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    gripper = None
    if has_gripper_data and gc.get("enabled", False):
        try:
            from zero_franky.robotiq import RobotiqGripperProxy

            gripper = RobotiqGripperProxy(
                server_host=gc.host,
                server_port=int(gc.port),
                auto_activate=True,
            )
        except Exception as e:
            print(f"  [gripper] Failed to initialize: {e}")
            gripper = None

    joint_pos_ik = None
    if tracker_mode == "ik":
        q_seed, f_t_ee = ik_frame_from_episode(episode)
        if q_seed is None:
            reset_joint_config = cfg.teleop.get("reset_joint_config", None)
            if reset_joint_config is not None:
                q_seed = np.asarray(reset_joint_config, dtype=float)
        ik = CartesianIK(
            f_t_ee=f_t_ee,
            joint_limit_margin=float(rc.get("ik_joint_limit_margin", 0.02)),
            position_tolerance=float(rc.get("ik_position_tolerance", 1e-3)),
            rotation_tolerance=float(rc.get("ik_rotation_tolerance", 1e-2)),
        )
        print(f"  Solving IK for {n_steps} waypoints...")
        joint_pos_ik = ik.solve_trajectory(
            ee_pos,
            ee_rot,
            q_seed=q_seed,
            timestamps=timestamps,
            max_distance=rc.get("ik_max_distance", 0.2),
        )
        print("  IK solve complete.")

    start_joint_pos = joint_pos[0] if joint_pos_ik is None else joint_pos_ik[0]
    print(f"  Pre-positioning to start configuration...")
    robot.move(JointMotion(
        JointState(start_joint_pos),
        relative_dynamics_factor=0.1,
    ))

    gripper_open_for_record = None
    if gripper is not None:
        initial_gripper_open = bool(round(float(gripper_open_data[0])))
        initial_width = (
            gc.get("open_width_m", 0.085) if initial_gripper_open else gc.get("close_width_m", 0.0)
        )
        gripper.move_width(
            initial_width,
            speed=int(gc.get("speed", 255)),
            force=int(gc.get("force", 255)),
            wait=True,
            max_width_m=gc.get("max_width_m", 0.085),
        )
        gripper_open_for_record = initial_gripper_open

    time.sleep(0.5)

    vc = cfg.get("visualization", {})
    pointcloud_source, pc, pointcloud_enabled = _resolve_pointcloud_source(cfg)

    cameras = {}
    recorder = None
    record_enabled = bool(rc.get("record", False))
    # Cameras are needed for recording and/or for live point-cloud visualization.
    if record_enabled or pointcloud_enabled:
        from clear_franka.camera import enabled_camera_names, make_zed_camera
        include = (pointcloud_source,) if pointcloud_enabled else ()
        for name in enabled_camera_names(cfg, include=include):
            cameras[name] = make_zed_camera(cfg, name)
            cameras[name].run()

    if record_enabled:
        extrinsics_metadata = {}
        for cam_name in cameras:
            ext_path = vc.get("pointclouds", {}).get(cam_name, {}).get(
                "extrinsics_path", f"./data/extrinsics_{cam_name}.json"
            )
            try:
                with resolve_project_path(ext_path).open("r") as f:
                    extrinsics_metadata[f"extrinsics_{cam_name}"] = f.read()
            except OSError:
                pass

        from clear_franka.recorder import TrajectoryRecorder
        recorder = TrajectoryRecorder(
            save_dir=cfg.data_dir,
            cameras=cameras,
            metadata={
                "replay_episode": str(episode_path),
                "replay_speed": float(rc.speed),
                "replay_tracker": tracker_mode,
                "gripper_enabled": gripper is not None,
                **extrinsics_metadata,
            },
        )
        recorder.start()

    visualizer, pointcloud_camera, pointcloud_frame_name = _setup_replay_visualizer(
        cfg, cameras, pointcloud_source, pc
    )
    if visualizer is not None:
        # Overlay the deploy safety box so it's obvious when a replayed trajectory
        # leaves it. Replay does NOT clip to this box (only deploy does), so the
        # arm can — and will — be commanded outside it if the recording goes there.
        dep = cfg.get("deploy", {})
        ws_lo = dep.get("workspace_lo", None)
        ws_hi = dep.get("workspace_hi", None)
        if ws_lo is not None and ws_hi is not None:
            visualizer.add_workspace_box(ws_lo, ws_hi)
            print(
                f"  [viser] deploy.workspace box overlaid: lo={list(ws_lo)} hi={list(ws_hi)}"
            )
        # Overlay the preprocess-pipeline EE paths: original (recorded) vs
        # TOPPRA-retimed vs Ruckig-smoothed (the executed path).
        if ee_stages:
            stage_colors = {
                "original": (180, 180, 180),   # grey — raw recorded
                "retimed": (0, 210, 255),      # cyan — TOPPRA retimed (pre-smooth)
                "smoothed": (255, 0, 170),     # magenta — Ruckig smoothed (executed)
            }
            for stage, col in stage_colors.items():
                path = ee_stages.get(stage)
                if path is not None and len(path) >= 2:
                    visualizer.add_path_overlay(f"/replay_traj/{stage}", path, color=col)
                    print(f"  [viser] {stage} EE path overlaid ({len(path)} pts)")
    viz = (
        _ReplayVizUpdater(
            visualizer, robot, pointcloud_camera, pointcloud_frame_name, pc,
            float(vc.get("update_hz", 30.0)),
        )
        if visualizer is not None
        else None
    )

    try:
        play_fn = play_cartesian_trajectory if tracker_mode == "cartesian" else play_joint_trajectory
        play_joint_pos = joint_pos if joint_pos_ik is None else joint_pos_ik
        play_joint_vel = joint_vel if joint_pos_ik is None else np.zeros_like(joint_pos_ik)
        play_has_joint_vel = has_joint_vel if joint_pos_ik is None else True
        common_kwargs = dict(
            robot=robot,
            rc=rc,
            gc=gc,
            timestamps=timestamps,
            gripper=gripper,
            gripper_open_data=gripper_open_data,
            recorder=recorder,
            gripper_open_for_record=gripper_open_for_record,
            viz=viz,
        )
        if tracker_mode == "cartesian":
            common_kwargs.update(
                cfg=cfg,
                ee_pos=ee_pos,
                ee_rot=ee_rot,
                joint_pos=joint_pos,
            )
        else:
            common_kwargs.update(
                stiffness=stiffness,
                joint_pos=play_joint_pos,
                joint_vel=play_joint_vel,
                has_joint_vel=play_has_joint_vel,
            )
        gripper_open_for_record = play_with_recovery(play_fn=play_fn, **common_kwargs)
        if recorder is not None:
            recorder.close()
            recorder = None

        if prompt_reverse_reset():
            reverse_timestamps = timestamps[-1] - timestamps[::-1]
            reverse_joint_pos = play_joint_pos[::-1]
            reverse_joint_vel = (
                -play_joint_vel[::-1] if play_has_joint_vel else play_joint_vel[::-1]
            )
            reverse_ee_pos = ee_pos[::-1] if ee_pos is not None else None
            reverse_ee_rot = ee_rot[::-1] if ee_rot is not None else None
            reverse_gripper_open_data = (
                gripper_open_data[::-1] if gripper_open_data is not None else gripper_open_data
            )
            reverse_kwargs = dict(
                robot=robot,
                rc=rc,
                gc=gc,
                timestamps=reverse_timestamps,
                gripper=gripper,
                gripper_open_data=reverse_gripper_open_data,
                gripper_open_for_record=gripper_open_for_record,
                viz=viz,
            )
            if tracker_mode == "cartesian":
                reverse_kwargs.update(
                    cfg=cfg,
                    ee_pos=reverse_ee_pos,
                    ee_rot=reverse_ee_rot,
                    joint_pos=reverse_joint_pos,
                )
            else:
                reverse_kwargs.update(
                    stiffness=stiffness,
                    joint_pos=reverse_joint_pos,
                    joint_vel=reverse_joint_vel,
                    has_joint_vel=play_has_joint_vel,
                )
            play_with_recovery(play_fn=play_fn, **reverse_kwargs)
            print("  Reverse reset complete.")
        else:
            print("  Reverse reset skipped.")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
    finally:
        robot.stop_state_stream()
        if recorder is not None:
            recorder.close()
        if gripper is not None:
            gripper.disconnect()
        for cam in cameras.values():
            cam.close()
