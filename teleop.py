"""Teleoperate the robot with a 3Dconnexion SpaceMouse in Cartesian impedance mode.

Push/twist the SpaceMouse knob to stream velocity commands to the end-effector.
Tap the left button to toggle motion on/off.
Tap the right button to toggle the gripper open/closed.
Tap both buttons together to start/stop recording.
Press Ctrl-C to stop.
"""

import threading

import numpy as np
from omegaconf import DictConfig

from threed_mouse import ThreeDMouse
from threed_mouse.geometry import pack_Rp, so3_exp
from threed_mouse.threedmousefilter import ThreeDMouseFilter

from clear_franka.recorder import TrajectoryRecorder
from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
from clear_franka.utils import LoopRatePrinter

DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


class AsyncTargetSender:
    def __init__(self, tracker, affine_cls, twist_cls):
        self._tracker = tracker
        self._affine_cls = affine_cls
        self._twist_cls = twist_cls
        self._condition = threading.Condition()
        self._latest = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, target_rot, target_pos, v_world, w_world) -> None:
        with self._condition:
            self._latest = (
                np.asarray(target_rot, dtype=float).copy(),
                np.asarray(target_pos, dtype=float).copy(),
                np.asarray(v_world, dtype=float).copy(),
                np.asarray(w_world, dtype=float).copy(),
            )
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                target_rot, target_pos, v_world, w_world = self._latest
                self._latest = None

            try:
                pose = self._affine_cls(pack_Rp(target_rot, target_pos))
                twist = self._twist_cls(v_world, w_world)
                self._tracker.set_target(pose, twist)
            except Exception as exc:
                print(f"\n  [tracker] async set_target failed: {exc}")


def run_teleop(cfg: DictConfig):
    from net_franky.franky import Affine, CartesianImpedanceTracker, ControlException, Robot, Twist

    tc = cfg.teleop
    sc = tc.spacemouse
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

    cameras = {}
    pointcloud_camera = None
    if pointcloud_enabled or any(
        cfg.get("cameras", {}).get(n, {}).get("enabled", False)
        for n in ("third_person", "hand")
    ):
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

    recorder = TrajectoryRecorder(
        save_dir=cfg.data_dir,
        metadata={
            "linear_scale": tc.linear_scale,
            "angular_scale": tc.angular_scale,
            "period": tc.period,
            "gripper_enabled": bool(gc.get("enabled", False)),
        },
        cameras=cameras,
    )

    gripper = None
    gripper_open = bool(gc.get("initial_open", True))
    if gc.get("enabled", False):
        try:
            gripper = RobotiqGripperProxy(
                server_host=gc.get("host", cfg.net_franky.ip),
                server_port=int(gc.get("port", cfg.net_franky.port)),
                com_port=gc.get("com_port", "auto"),
                device_id=int(gc.get("device_id", 9)),
                connection_type=gc.get("connection_type", "RTU"),
                tcp_host=gc.get("tcp_host", "127.0.0.1"),
                tcp_port=int(gc.get("tcp_port", 54321)),
                auto_activate=bool(gc.get("activate_on_start", True)),
            )
            if gc.get("activate_on_start", True):
                print("Robotiq gripper activated.")
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
    print("  Tap LEFT button to toggle motion on/off.")
    if gripper is not None:
        print("  Tap RIGHT button to toggle gripper open/closed.")
    else:
        print("  RIGHT button gripper control disabled.")
    print("  Tap LEFT+RIGHT together to start/stop recording.")
    print("  Press Ctrl-C to stop.")

    loop_rate = LoopRatePrinter()
    try:
        while True:
            robot.recover_from_errors()
            enabled = False
            prev_button = 0
            prev_right_button = 0
            prev_record_button = 0
            last_pointcloud_timestamp = None

            try:
                with CartesianImpedanceTracker(
                    robot,
                    translational_stiffness=tc.translational_stiffness,
                    rotational_stiffness=tc.rotational_stiffness,
                    nullspace_stiffness=tc.nullspace_stiffness,
                    lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
                    upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
                    period=tc.period,
                ) as tracker:
                    initial_pose = tracker.current_pose.end_effector_pose
                    target_pos = np.asarray(initial_pose.translation, dtype=float)
                    target_rot = np.asarray(initial_pose.matrix[:3, :3], dtype=float)
                    target_sender = AsyncTargetSender(tracker, Affine, Twist)

                    try:
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

                            if record_button and not prev_record_button:
                                loop_rate.newline()
                                recorder.toggle()
                            elif button and not prev_button:
                                enabled = not enabled
                                loop_rate.newline()
                                print("  ENABLED" if enabled else "  DISABLED")
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

                                target_sender.send(
                                    target_rot,
                                    target_pos,
                                    v_world,
                                    w_world,
                                )

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
                    finally:
                        target_sender.close()
            except ControlException as e:
                loop_rate.newline()
                print(f"\n  Controller faulted: {e}")
                print("  Recovering... tap button to re-enable.")
    finally:
        loop_rate.newline()
        recorder.close()
        if gripper is not None:
            gripper.disconnect()
        for cam in cameras.values():
            cam.stop_pointcloud_stream()
        mouse.close()
        for cam in cameras.values():
            cam.close()
