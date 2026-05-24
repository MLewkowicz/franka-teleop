"""Teleoperate the robot with a 3Dconnexion SpaceMouse in Cartesian impedance mode.

Push/twist the SpaceMouse knob to stream velocity commands to the end-effector.
Tap the left button to toggle motion on/off.
Hold the left button to reset to the configured start joint config.
Tap the right button to toggle the gripper open/closed.
Tap both buttons together to start/stop recording.
Press Ctrl-C to stop.
"""

import contextlib
import threading
import time

import numpy as np
from omegaconf import DictConfig

from threed_mouse import ThreeDMouse
from threed_mouse.geometry import pack_Rp, so3_exp
from threed_mouse.threedmousefilter import ThreeDMouseFilter

from clear_franka.recorder import TrajectoryRecorder
from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
from clear_franka.utils import LoopRatePrinter

from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS

RESET_LONG_PRESS_S = 0.8



def run_teleop(cfg: DictConfig):
    from zero_franky import Robot
    from franky import Affine, JointMotion, JointState, JointStopMotion, Twist

    tc = cfg.teleop
    sc = tc.spacemouse
    reset_joint_config = np.asarray(tc.reset_joint_config, dtype=float)
    gc = cfg.get("gripper", {})
    vc = cfg.get("visualization", {}).get("viser", {})
    pointcloud_cfg = vc.get("pointclouds", {})
    pointcloud_source = next(
        (name for name in ("third_person", "hand") if pointcloud_cfg.get(name, {}).get("enabled", False)),
        "third_person",
    )
    pc = pointcloud_cfg.get(pointcloud_source, {})
    pointcloud_enabled = bool(vc.get("enabled", False) and pc.get("enabled", False))

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    mouse = ThreeDMouse(control_rate=sc.control_rate)
    mouse._frame_rotation_linear = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ], dtype=float)
    mouse._frame_rotation_angular = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ], dtype=float)
    mouse.run()

    input_filter = ThreeDMouseFilter(
        smoothing_factor=sc.smoothing_factor,
        softmax_temp=sc.softmax_temp,
        translation_modifier=tc.linear_scale,
        rotation_modifer=tc.angular_scale,
        translation_deadband=sc.translation_deadband,
        rotation_deadband=sc.rotation_deadband,
        translation_enabled=True,
        rotation_enabled=True,
    )

    record_mode = str(tc.get("record", "joints"))
    record_cameras = record_mode == "all"

    cameras = {}
    pointcloud_camera = None
    if pointcloud_enabled or (record_cameras and any(
        cfg.get("cameras", {}).get(n, {}).get("enabled", False)
        for n in ("third_person", "hand")
    )):
        try:
            from clear_franka.camera import enabled_camera_names, get_camera_config, make_zed_camera

            include = (pointcloud_source,) if pointcloud_enabled else ()
            camera_names = enabled_camera_names(cfg, include=include)
            missing_serial = [
                name for name in camera_names
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

    latest_joint_pos = None

    extrinsics_metadata = {}
    for cam_name in cameras:
        cam_pc_cfg = vc.get("pointclouds", {}).get(cam_name, {})
        ext_path = cam_pc_cfg.get("extrinsics_path", f"./data/extrinsics_{cam_name}.json")
        try:
            with open(ext_path) as f:
                extrinsics_metadata[f"extrinsics_{cam_name}"] = f.read()
        except OSError:
            print(f"  [recorder] No extrinsics found for {cam_name} at {ext_path}; not embedded in episode.")

    recorder_cfg = cfg.get("recorder", {})
    recorder = TrajectoryRecorder(
        save_dir=cfg.data_dir,
        metadata={
            "linear_scale": tc.linear_scale,
            "angular_scale": tc.angular_scale,
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
        try:
            gripper = RobotiqGripperProxy(
                server_host=gc.host,
                server_port=int(gc.port),
                com_port=gc.com_port,
                device_id=int(gc.device_id),
                connection_type=gc.connection_type,
                tcp_host=gc.tcp_host,
                tcp_port=int(gc.tcp_port),
                auto_activate=bool(gc.activate_on_start),
            )
            print("Robotiq gripper proxy ready.")
        except Exception as e:
            print(f"  [gripper] Failed to initialize: {e}")
            raise

    visualizer = None
    camera_frame_added = False
    if vc.get("enabled", False):
        try:
            from clear_franka.visualization import CortadoViserVisualizer

            visualizer = CortadoViserVisualizer(
                host=vc.get("host", "0.0.0.0"),
                port=int(vc.get("port", 8080)),
            )
            visualizer.update_gripper_width(
                gc.get("open_width_m", 0.085) if gripper_open else gc.get("close_width_m", 0.0),
                max_width_m=gc.get("max_width_m", 0.085),
            )
            if pointcloud_enabled and pointcloud_camera is not None:
                frame_name = pc.get("frame_name", "/third_person_zed")
                if pointcloud_source == "hand":
                    visualizer.add_hand_camera_frame_from_extrinsics(
                        frame_name,
                        pc.get("extrinsics_path", "./data/extrinsics_hand.json"),
                    )
                else:
                    visualizer.add_camera_frame_from_extrinsics(
                        frame_name,
                        pc.get("extrinsics_path", "./data/extrinsics_third_person.json"),
                    )
                camera_frame_added = True
                pointcloud_camera.start_pointcloud_stream(
                    update_hz=float(pc.get("update_hz", 5.0)),
                    stride=int(pc.get("stride", 4)),
                    max_points=int(pc.get("max_points", 100_000)),
                    max_distance_m=float(pc.get("max_distance_m", 3.0)),
                )
                print(f"  [viser] {pointcloud_source} ZED point cloud enabled.")
        except Exception as e:
            print(f"  [viser] Failed to initialize: {e}")
            if vc.get("required", False):
                raise
            print("  Continuing without viser.")
            visualizer = None
            pointcloud_enabled = False

    print("SpaceMouse teleop ready.")
    print("  Tap LEFT to toggle motion on/off.")
    print(f"  Hold LEFT ({RESET_LONG_PRESS_S}s) to reset to start joint config.")
    if gripper is not None:
        print("  Tap RIGHT to toggle gripper open/closed.")
    else:
        print("  RIGHT button gripper control disabled.")
    print("  Tap LEFT+RIGHT together to start/stop recording.")
    print("  Press Ctrl-C to stop.")

    loop_rate = LoopRatePrinter()
    reset_pending = False
    with contextlib.ExitStack() as stack:
        for cam in cameras.values():
            stack.enter_context(cam)
        stack.callback(mouse.close)
        if gripper is not None:
            stack.enter_context(gripper)
        stack.enter_context(recorder)
        stack.callback(loop_rate.newline)
        while True:
            robot.recover_from_errors()
            # Drain any motions that might be left over
            robot.join_motion(2)

            if reset_pending:
                reset_pending = False
                print("  Resetting to start config (release LEFT to stop)...")
                robot.move(JointMotion(
                    JointState(reset_joint_config),
                    relative_dynamics_factor=float(tc.get("reset_dynamics_factor", 0.1)),
                ), asynchronous=True)
                motion_done = threading.Event()
                threading.Thread(
                    target=lambda: (robot.join_motion(), motion_done.set()),
                    daemon=True,
                ).start()
                stopped_early = False
                while not motion_done.is_set():
                    sample = mouse.get_controller_state()
                    if sample is not None:
                        buttons = np.asarray(sample.buttons, dtype=int)
                        cur_button = int(buttons[0]) if len(buttons) > 0 else 1
                    else:
                        cur_button = 1
                    if not cur_button:
                        robot.move(JointStopMotion())
                        stopped_early = True
                        break
                    time.sleep(0.01)
                motion_done.wait()
                print("  Reset stopped." if stopped_early else "  Reset complete.")

            enabled = False
            prev_button = 0
            prev_right_button = 0
            prev_record_button = 0
            last_pointcloud_timestamp = None
            left_press_time = None
            left_used_in_record = False

            with robot.start_cartesian_impedance_session(
                period=0.001,
                translational_stiffness=tc.translational_stiffness,
                rotational_stiffness=tc.rotational_stiffness,
                nullspace_stiffness=tc.nullspace_stiffness,
                lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
                upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
            ) as session:
                try:
                    teleop_state = robot.get_last_teleop_state()
                    initial_pose = np.asarray(teleop_state["O_T_EE"], dtype=float)
                    target_pos = initial_pose[:3, 3].copy()
                    target_rot = initial_pose[:3, :3].copy()
                    while True:
                        loop_rate.start_tick()

                        sample = mouse.get_controller_state()
                        if sample is None:
                            loop_rate.finish_tick()
                            continue

                        buttons = np.asarray(sample.buttons, dtype=int)
                        button = int(buttons[0]) if len(buttons) > 0 else 0
                        right_button = int(buttons[1]) if len(buttons) > 1 else 0
                        record_button = button and right_button

                        # Track left-alone press start for long-press detection
                        if button and not prev_button and not right_button:
                            left_press_time = time.monotonic()
                            left_used_in_record = False
                        # Cancel long-press if right pressed while left held
                        if right_button and not prev_right_button and button:
                            left_used_in_record = True
                        # Long-press threshold crossed → trigger reset immediately
                        if button and left_press_time is not None and not left_used_in_record:
                            if time.monotonic() - left_press_time >= RESET_LONG_PRESS_S:
                                enabled = False
                                reset_pending = True

                        if record_button and not prev_record_button:
                            left_used_in_record = True
                            loop_rate.newline()
                            recorder.toggle()
                        elif not button and prev_button:
                            # Left released — always a short tap if we get here
                            if not left_used_in_record and left_press_time is not None:
                                enabled = not enabled
                                if enabled:
                                    robot.recover_from_errors()
                                loop_rate.newline()
                                print("  ENABLED" if enabled else "  DISABLED")
                            left_press_time = None
                            left_used_in_record = False
                        elif right_button and not prev_right_button:
                            if gripper is not None:
                                target_width = (
                                    gc.get("close_width_m", 0.0)
                                    if gripper_open
                                    else gc.get("open_width_m", 0.085)
                                )
                                try:
                                    gripper.move_width(
                                        target_width,
                                        speed=int(gc.get("speed", 255)),
                                        force=int(gc.get("force", 255)),
                                        wait=False,
                                        max_width_m=gc.get("max_width_m", 0.085),
                                    )
                                    gripper_open = not gripper_open
                                    if visualizer is not None:
                                        visualizer.update_gripper_width(
                                            target_width,
                                            max_width_m=gc.get("max_width_m", 0.085),
                                        )
                                    state = "open" if gripper_open else "closed"
                                    loop_rate.newline()
                                    print(f"  [gripper] toggled {state}")
                                except Exception as e:
                                    loop_rate.newline()
                                    print(f"  [gripper] toggle failed: {e}")
                        prev_button = button
                        prev_right_button = right_button
                        prev_record_button = record_button

                        teleop_state = robot.get_last_teleop_state()
                        joint_pos = np.asarray(teleop_state["q"], dtype=float)
                        latest_joint_pos = joint_pos.copy()
                        joint_vel = np.asarray(teleop_state["dq"], dtype=float)
                        measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                        robot_abs_time = float(teleop_state["abs_time"])
                        robot_pos = measured_pose[:3, 3]
                        robot_rot = measured_pose[:3, :3]

                        if visualizer is not None:
                            visualizer.update(joint_pos)
                            visualizer.update_eef_frame(measured_pose)
                            if pointcloud_enabled and pointcloud_camera is not None:
                                latest_pointcloud = pointcloud_camera.get_latest_pointcloud()
                                if latest_pointcloud is not None:
                                    points, colors, timestamp = latest_pointcloud
                                    if timestamp != last_pointcloud_timestamp:
                                        visualizer.update_pointcloud(
                                            pc.get(
                                                "frame_name",
                                                "/cortado/fr3_link8/hand_zed"
                                                if pointcloud_source == "hand"
                                                else "/third_person_zed",
                                            ),
                                            points,
                                            colors,
                                            point_size=float(pc.get("point_size", 0.01)),
                                        )
                                        last_pointcloud_timestamp = timestamp

                        if enabled:
                            v = np.asarray(sample.xyz, dtype=float).copy()
                            w = np.asarray(sample.rpy, dtype=float).copy()
                            input_filter._translation_modifier(v)
                            input_filter._rotation_modifier(w)

                            if tc.global_frame:
                                target_pos = robot_pos + v
                                target_rot = so3_exp(w) @ robot_rot
                                v_world = v
                                w_world = w
                            else:
                                base_rot = robot_rot
                                target_pos = robot_pos + base_rot @ v
                                target_rot = base_rot @ so3_exp(w)
                                v_world = base_rot @ v
                                w_world = base_rot @ w

                            try:
                                session.set_cartesian_reference(
                                    Affine(pack_Rp(target_rot, target_pos)),
                                    Twist(v_world, w_world),
                                )
                            except Exception as exc:
                                print(f"\n  [tracker] set_cartesian_reference failed: {exc}")

                        recorder.step(
                            ee_pos=robot_pos,
                            ee_rot=robot_rot,
                            cmd_linear_vel=v_world if enabled else np.zeros(3),
                            cmd_angular_vel=w_world if enabled else np.zeros(3),
                            buttons=button | (right_button << 1),
                            enabled=enabled,
                            joint_pos=joint_pos,
                            joint_vel=joint_vel,
                            gripper_open=gripper_open if gripper is not None else None,
                            robot_abs_time=robot_abs_time,
                        )
                        loop_rate.finish_tick()
                        if reset_pending:
                            break
                except RuntimeError as e:
                    loop_rate.newline()
                    print(f"\n  Controller faulted: {e}")
                    print("  Recovering... tap button to re-enable.")
