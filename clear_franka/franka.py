"""Shared Franka hardware constants and runtime helpers."""

import time

DEFAULT_LOWER_JOINT_LIMITS = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
DEFAULT_UPPER_JOINT_LIMITS = [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]


def joint_friction_kwargs(cfg) -> dict:
    friction_cfg = cfg.get("joint_friction", {})
    if not friction_cfg or not friction_cfg.get("enabled", False):
        return {}

    kwargs = {
        "friction_coulomb": [float(v) for v in friction_cfg.coulomb],
        "friction_viscous": [float(v) for v in friction_cfg.viscous],
    }
    if friction_cfg.get("max_torque") is not None:
        kwargs["friction_max_torque"] = [float(v) for v in friction_cfg.max_torque]
    if friction_cfg.get("velocity_epsilon") is not None:
        kwargs["friction_velocity_epsilon"] = float(friction_cfg.velocity_epsilon)
    return kwargs


def wait_for_motion_idle(robot, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if robot.join_motion(min(poll_s, remaining)):
            return True


def stop_tracker_motion(robot, session, join_timeout: float = 1.0, idle_timeout_s: float = 5.0) -> bool:
    session.stop(join_timeout=join_timeout)
    return wait_for_motion_idle(robot, timeout_s=idle_timeout_s)
