"""Capture a 'home' joint config by hand-guiding the arm, then printing it.

Runs the same backdrivable joint-impedance float as demonstrate (zero stiffness
+ friction compensation), so you can physically move the arm to the desired pose
and read off the 7 joint angles — ready to paste into demonstrate.home_configs or
reset_joint_config in conf/config.yaml.

Run:  python capture_home_config.py
Hand-guide the arm, press Enter to print the current config, Ctrl-C to quit.
No recording, no cameras.
"""

import numpy as np
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from zero_franky import Robot, setup_zero_franky
    from zero_franky.tracker_policies import hold_current_joint
    from clear_franka.franka import (
        DEFAULT_LOWER_JOINT_LIMITS,
        DEFAULT_UPPER_JOINT_LIMITS,
        joint_friction_kwargs,
        stop_tracker_motion,
    )

    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port, pub_port=cfg.zero_franky.pub_port)
    dc = cfg.get("demonstrate", {})
    joint_stiffness = [float(v) for v in dc.get("joint_stiffness", [0.0] * 7)]

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    try:
        robot.join_motion(2)  # drain any leftover motion so the float session starts cleanly
    except Exception:
        pass

    session = robot.start_joint_impedance_session(
        hold_current_joint,
        period=0.001,
        stiffness=joint_stiffness,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        **joint_friction_kwargs(dc),
    )
    try:
        print("\nArm is in float mode — hand-guide it to the desired home pose.")
        print("Press Enter to capture the current joint config; Ctrl-C to quit.\n")
        while True:
            try:
                input("  [capture] Enter = print config > ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            try:
                state = robot.get_last_teleop_state()
            except RuntimeError:
                print("  (no state yet — move the arm slightly and retry)")
                continue
            q = np.asarray(state["q"], dtype=float)
            ee = np.asarray(state["O_T_EE"], dtype=float).reshape(4, 4)
            print("\n  reset_joint_config / home_configs entry (paste into conf/config.yaml):")
            print("  [" + ", ".join(f"{v:.6f}" for v in q) + "]")
            print(f"  (EE position xyz [m] = {np.array2string(ee[:3, 3], precision=4)})\n")
    finally:
        stop_tracker_motion(robot, session, join_timeout=1.0, idle_timeout_s=2.0)


if __name__ == "__main__":
    main()
