"""Kinesthetic demonstration recording with Franka Desk Pilot buttons.

The arm runs in joint impedance mode so an operator can physically guide it.
Press CHECK on the Desk Pilot to toggle the gripper.
Press CIRCLE on the Desk Pilot to start/stop recording.
Press CROSS on the Desk Pilot to reset when not recording.
Press Ctrl-C to stop.
"""

import contextlib
import json
import os
import time

import hydra
import numpy as np
from franky import JointMotion, JointState, PilotButton, RobotWebSession
from omegaconf import DictConfig

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    joint_friction_kwargs,
    stop_tracker_motion,
    wait_for_motion_idle,
)
from clear_franka.recorder import TrajectoryRecorder
from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
from clear_franka.utils import LoopRatePrinter, announce


def _desk_credentials(cfg: DictConfig) -> tuple[str, str, str]:
    desk_cfg = cfg.get("desk", {})
    hostname = str(desk_cfg.get("hostname", cfg.robot.ip))
    username = desk_cfg.get("username") or os.environ.get("FRANKA_DESK_USERNAME")
    password = desk_cfg.get("password") or os.environ.get("FRANKA_DESK_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set desk.username/desk.password in Hydra config or "
            "FRANKA_DESK_USERNAME/FRANKA_DESK_PASSWORD in the environment."
        )
    return hostname, str(username), str(password)


def _toggle_gripper(gripper, gripper_open: bool, gripper_cfg, visualizer=None) -> bool:
    target_width = (
        gripper_cfg.get("close_width_m", 0.0)
        if gripper_open
        else gripper_cfg.get("open_width_m", 0.085)
    )
    gripper.move_width(
        target_width,
        speed=int(gripper_cfg.get("speed", 255)),
        force=int(gripper_cfg.get("force", 255)),
        wait=False,
        max_width_m=gripper_cfg.get("max_width_m", 0.085),
    )
    if visualizer is not None:
        visualizer.update_gripper_width(target_width, max_width_m=gripper_cfg.get("max_width_m", 0.085))
    return not gripper_open


def _setup_cameras(cfg: DictConfig, record_cameras: bool, pointcloud_enabled: bool, pointcloud_source: str):
    cameras = {}
    pointcloud_camera = None
    if pointcloud_enabled or (
        record_cameras
        and any(cfg.get("cameras", {}).get(n, {}).get("enabled", False) for n in ("third_person", "hand"))
    ):
        try:
            from clear_franka.camera import enabled_camera_names, get_camera_config, make_zed_camera

            include = (pointcloud_source,) if pointcloud_enabled else ()
            camera_names = enabled_camera_names(cfg, include=include)
            missing_serial = [
                name
                for name in camera_names
                if len(camera_names) > 1 and get_camera_config(cfg, name)["serial_number"] is None
            ]
            if missing_serial:
                raise ValueError(
                    "Multiple ZED cameras are enabled; set camera.cameras.<name>.serial_number for "
                    + ", ".join(missing_serial)
                )
            for name in camera_names:
                cameras[name] = make_zed_camera(cfg, name)
                cameras[name].run()
            pointcloud_camera = cameras.get(pointcloud_source)
            print(f"  ZED cameras ready: {', '.join(cameras) if cameras else 'none'}.")
        except Exception as e:
            print(f"  [camera] Failed to initialize: {e}")
            print("  Continuing without camera.")
            for cam in cameras.values():
                cam.close()
            cameras = {}
            pointcloud_camera = None
            pointcloud_enabled = False
    return cameras, pointcloud_camera, pointcloud_enabled


def _metadata_for_cameras(cameras: dict, cfg: DictConfig, vc) -> dict:
    metadata = {}
    for cam_name in cameras:
        ext_path = cfg.get("cameras", {}).get(cam_name, {}).get(
            "extrinsics_path",
            f"./data/extrinsics_{cam_name}.json",
        )
        try:
            with open(ext_path) as f:
                metadata[f"extrinsics_{cam_name}"] = f.read()
        except OSError:
            print(f"  [recorder] No extrinsics found for {cam_name} at {ext_path}; not embedded in episode.")
    return metadata


def run_demonstrate(cfg: DictConfig):
    from zero_franky import Robot
    from zero_franky.tracker_policies import hold_current_joint

    gc = cfg.get("gripper", {})
    tc = cfg.get("teleop", {})
    dc = cfg.get("demonstrate", {})
    vc = cfg.get("visualization", {})
    pointcloud_cfg = vc.get("pointclouds", {})
    pointcloud_source = next(
        (name for name in ("third_person", "hand") if pointcloud_cfg.get(name, {}).get("enabled", False)),
        "third_person",
    )
    pc = pointcloud_cfg.get(pointcloud_source, {})
    pointcloud_enabled = bool(vc.get("enabled", False) and pc.get("enabled", False))

    record_mode = str(dc.get("record", tc.get("record", "joints")))
    record_cameras = record_mode == "all"
    reset_joint_config = np.asarray(tc.reset_joint_config, dtype=float)
    joint_stiffness = dc.joint_stiffness
    period = 0.001
    button_timeout = float(dc.get("button_timeout", period))
    button_debounce_s = float(dc.button_debounce_s)

    hostname, username, password = _desk_credentials(cfg)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    cameras, pointcloud_camera, pointcloud_enabled = _setup_cameras(
        cfg,
        record_cameras=record_cameras,
        pointcloud_enabled=pointcloud_enabled,
        pointcloud_source=pointcloud_source,
    )
    extrinsics_metadata = _metadata_for_cameras(cameras, cfg, vc)

    recorder_cfg = cfg.get("recorder", {})
    recorder = TrajectoryRecorder(
        save_dir=cfg.data_dir,
        metadata={
            "control_mode": "joint_impedance_demonstration",
            "joint_stiffness": json.dumps([float(v) for v in joint_stiffness]),
            "gripper_enabled": bool(gc.get("enabled", False)),
            **extrinsics_metadata,
        },
        cameras=cameras if record_cameras else {},
        record_svo=bool(recorder_cfg.get("record_svo", False)),
        svo_compression=str(recorder_cfg.get("svo_compression", "H264")),
    )

    gripper = None
    gripper_open = bool(gc.get("initial_open", True))
    if gc.get("enabled", False):
        gripper = RobotiqGripperProxy(
            server_host=gc.host,
            server_port=int(gc.port),
            com_port=gc.com_port,
            device_id=int(gc.device_id),
            connection_type=gc.connection_type,
            tcp_host=gc.tcp_host,
            tcp_port=int(gc.tcp_port),
            auto_activate=True,
        )
        print("Robotiq gripper proxy ready.")

    visualizer = None
    pointcloud_frame_name = None
    if vc.get("enabled", False):
        from clear_franka.visualization import CortadoViserVisualizer

        visualizer = CortadoViserVisualizer(host=vc.get("host", "0.0.0.0"), port=int(vc.get("port", 8080)))
        visualizer.update_gripper_width(
            gc.get("open_width_m", 0.085) if gripper_open else gc.get("close_width_m", 0.0),
            max_width_m=gc.get("max_width_m", 0.085),
        )
        if pointcloud_enabled and pointcloud_camera is not None:
            frame_name = pc.get(
                "frame_name",
                "hand_zed" if pointcloud_source == "hand" else "/third_person_zed",
            )
            if pointcloud_source == "hand":
                pointcloud_frame_name = visualizer.add_hand_camera_frame_from_extrinsics(
                    frame_name,
                    cfg.cameras.hand.get("extrinsics_path", "./data/extrinsics_hand.json"),
                )
            else:
                pointcloud_frame_name = visualizer.add_camera_frame_from_extrinsics(
                    frame_name,
                    cfg.cameras.third_person.get(
                        "extrinsics_path",
                        "./data/extrinsics_third_person.json",
                    ),
                )
            pointcloud_camera.start_pointcloud_stream(
                update_hz=float(pc.get("update_hz", 5.0)),
                stride=int(pc.get("stride", 4)),
                max_points=int(pc.get("max_points", 100_000)),
                max_distance_m=float(pc.get("max_distance_m", 3.0)),
            )
            print(f"  [viser] {pointcloud_source} ZED point cloud enabled.")


    print("Joint impedance demonstration ready.")
    print("  Physically guide the arm to demonstrate.")
    print("  Press CHECK to toggle the gripper.")
    print("  Press CIRCLE to start/stop recording.")
    print("  Press CROSS to reset to the start config when not recording.")
    print("  Press Ctrl-C to stop.")

    loop_rate = LoopRatePrinter()
    latest_pointcloud_timestamp = None
    pressed_buttons: set[PilotButton] = set()
    last_button_action = {
        PilotButton.CIRCLE: 0.0,
        PilotButton.CHECK: 0.0,
        PilotButton.CROSS: 0.0,
    }
    reset_pending = False
    suppress_cross_until_release = False
    with contextlib.ExitStack() as stack:
        for cam in cameras.values():
            stack.enter_context(cam)
        if gripper is not None:
            stack.enter_context(gripper)
        stack.enter_context(recorder)
        web = stack.enter_context(RobotWebSession(hostname, username, password, token_storage=True))
        stack.callback(loop_rate.newline)

        while True:
            if reset_pending:
                reset_pending = False
                pressed_buttons.clear()
                loop_rate.newline()
                print("  Resetting to start config...")
                robot.recover_from_errors()
                if not wait_for_motion_idle(robot, timeout_s=5.0):
                    print("  [reset] previous motion did not report idle; trying reset anyway.")
                robot.move(
                    JointMotion(
                        JointState(reset_joint_config),
                        relative_dynamics_factor=0.1,
                    ),
                    asynchronous=True,
                )
                if not wait_for_motion_idle(robot, timeout_s=30.0):
                    raise TimeoutError("Timed out waiting for reset motion to finish.")
                print("  Reset complete.")
                suppress_cross_until_release = True

            session = robot.start_joint_impedance_session(
                hold_current_joint,
                period=period,
                stiffness=[float(v) for v in joint_stiffness],
                lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
                upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
                **joint_friction_kwargs(dc),
            )
            session_stopped = False
            try:
                while True:
                    loop_rate.start_tick()
                    for event in web.poll_buttons(timeout=button_timeout):
                        if event.button == PilotButton.CROSS and suppress_cross_until_release:
                            if not event.pressed:
                                suppress_cross_until_release = False
                                pressed_buttons.discard(event.button)
                            continue

                        if not event.pressed:
                            pressed_buttons.discard(event.button)
                            continue

                        already_pressed = event.button in pressed_buttons
                        pressed_buttons.add(event.button)
                        if already_pressed:
                            continue

                        now = time.monotonic()
                        if now - last_button_action.get(event.button, 0.0) < button_debounce_s:
                            continue

                        if event.button == PilotButton.CIRCLE:
                            last_button_action[event.button] = now
                            loop_rate.newline()
                            recorder.toggle()
                            announce("recording stopped" if not recorder.recording else "recording started")
                        elif event.button == PilotButton.CHECK:
                            last_button_action[event.button] = now
                            loop_rate.newline()
                            if gripper is None:
                                print("  [gripper] CHECK ignored; gripper disabled.")
                                continue
                            try:
                                gripper_open = _toggle_gripper(gripper, gripper_open, gc, visualizer=visualizer)
                                print(f"  [gripper] toggled {'open' if gripper_open else 'closed'}")
                            except Exception as e:
                                print(f"  [gripper] toggle failed: {e}")
                        elif event.button == PilotButton.CROSS:
                            last_button_action[event.button] = now
                            loop_rate.newline()
                            if recorder.recording:
                                print("  [reset] ignored while recording.")
                            else:
                                reset_pending = True

                    if reset_pending:
                        loop_rate.finish_tick()
                        session_stopped = True
                        if not stop_tracker_motion(robot, session, join_timeout=5.0, idle_timeout_s=5.0):
                            print("  [reset] joint impedance motion did not report idle after stop.")
                        break


                    teleop_state = robot.get_last_teleop_state()
                    joint_pos = np.asarray(teleop_state["q"], dtype=float)
                    joint_vel = np.asarray(teleop_state["dq"], dtype=float)
                    measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                    robot_abs_time = float(teleop_state["abs_time"])

                    if visualizer is not None:
                        visualizer.update(joint_pos)
                        visualizer.update_eef_frame(measured_pose)
                        if pointcloud_enabled and pointcloud_camera is not None:
                            latest_pointcloud = pointcloud_camera.get_latest_pointcloud()
                            if latest_pointcloud is not None:
                                points, colors, timestamp = latest_pointcloud
                                if timestamp != latest_pointcloud_timestamp:
                                    visualizer.update_pointcloud(
                                        pointcloud_frame_name
                                        or pc.get("frame_name", "/third_person_zed"),
                                        points,
                                        colors,
                                        point_size=float(pc.get("point_size", 0.01)),
                                    )
                                    latest_pointcloud_timestamp = timestamp

                    recorder.step(
                        ee_pos=measured_pose[:3, 3],
                        ee_rot=measured_pose[:3, :3],
                        cmd_linear_vel=np.zeros(3),
                        cmd_angular_vel=np.zeros(3),
                        buttons=(
                            (1 if PilotButton.CIRCLE in pressed_buttons else 0)
                            | ((1 if PilotButton.CHECK in pressed_buttons else 0) << 1)
                            | ((1 if PilotButton.CROSS in pressed_buttons else 0) << 2)
                        ),
                        enabled=True,
                        joint_pos=joint_pos,
                        joint_vel=joint_vel,
                        gripper_open=gripper_open if gripper is not None else None,
                        robot_abs_time=robot_abs_time,
                    )
                    loop_rate.finish_tick()
            finally:
                if not session_stopped:
                    stop_tracker_motion(robot, session, join_timeout=1.0, idle_timeout_s=2.0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from zero_franky import setup_zero_franky

    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port, pub_port=cfg.zero_franky.pub_port)
    run_demonstrate(cfg)


if __name__ == "__main__":
    main()
