"""Teleoperate the robot with a 3Dconnexion SpaceMouse in Cartesian impedance mode.

Push/twist the SpaceMouse knob to stream velocity commands to the end-effector.
Tap the left button to toggle motion on/off.
Tap the right button to start/stop recording.
Press Ctrl-C to stop.
"""

import numpy as np
from omegaconf import DictConfig

from net_franky.franky import Affine, CartesianImpedanceTracker, ControlException, Robot, Twist
from threed_mouse import ThreeDMouse
from threed_mouse.geometry import pack_Rp, so3_exp
from threed_mouse.threedmousefilter import ThreeDMouseFilter

from recorder import TrajectoryRecorder

DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


def run_teleop(cfg: DictConfig):
    tc = cfg.teleop
    sc = tc.spacemouse

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

    camera = None
    if cfg.get("camera", {}).get("enabled", False):
        try:
            from camera import ZedCamera
            camera = ZedCamera(
                resolution=cfg.camera.resolution,
                fps=cfg.camera.fps,
                depth_mode=cfg.camera.depth_mode,
            )
            camera.run()
            print("  ZED 2i camera ready.")
        except Exception as e:
            print(f"  [camera] Failed to initialize: {e}")
            print("  Continuing without camera.")
            camera = None

    recorder = TrajectoryRecorder(
        save_dir=cfg.data_dir,
        metadata={
            "linear_scale": tc.linear_scale,
            "angular_scale": tc.angular_scale,
            "period": tc.period,
        },
        camera=camera,
    )

    print("SpaceMouse teleop ready.")
    print("  Tap LEFT button to toggle motion on/off.")
    print("  Tap RIGHT button to start/stop recording.")
    print("  Press Ctrl-C to stop.")

    try:
        while True:
            robot.recover_from_errors()
            enabled = False
            prev_button = 0
            prev_right_button = 0

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
                    while tracker.tick():
                        sample = mouse.get_controller_state()
                        if sample is None:
                            continue

                        buttons = np.asarray(sample.buttons, dtype=int)
                        button = int(buttons[0]) if len(buttons) > 0 else 0
                        if button and not prev_button:
                            enabled = not enabled
                            print("  ENABLED" if enabled else "  DISABLED")
                        prev_button = button

                        right_button = int(buttons[1]) if len(buttons) > 1 else 0
                        if right_button and not prev_right_button:
                            recorder.toggle()
                        prev_right_button = right_button

                        current = tracker.current_pose.end_effector_pose
                        robot_pos = np.asarray(current.translation, dtype=float)
                        robot_rot = np.asarray(current.matrix[:3, :3], dtype=float)

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
                                target_pos = robot_pos + robot_rot @ v
                                target_rot = robot_rot @ so3_exp(w)
                                v_world = robot_rot @ v
                                w_world = robot_rot @ w

                            pose = Affine(pack_Rp(target_rot, target_pos))
                            tracker.set_target(pose, Twist(v_world, w_world))
                        else:
                            tracker.set_target(current)

                        recorder.step(
                            ee_pos=robot_pos,
                            ee_rot=robot_rot,
                            cmd_linear_vel=v_world if enabled else np.zeros(3),
                            cmd_angular_vel=w_world if enabled else np.zeros(3),
                            buttons=button,
                            enabled=enabled,
                        )
            except ControlException as e:
                print(f"\n  Controller faulted: {e}")
                print("  Recovering... tap button to re-enable.")
    finally:
        recorder.close()
        mouse.close()
        if camera is not None:
            camera.close()
