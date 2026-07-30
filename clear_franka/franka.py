"""Shared Franka hardware constants and runtime helpers."""

import time

DEFAULT_LOWER_JOINT_LIMITS = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
DEFAULT_UPPER_JOINT_LIMITS = [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]


def joint_friction_kwargs(cfg, speed: float = 1.0) -> dict:
    """Build the franky ``friction=`` kwarg for a ``start_*_impedance_tracker`` call.

    franky takes a single ``FrictionCompensationParams`` (coulomb, viscous,
    max_torque, velocity_epsilon) instead of the old flat ``friction_coulomb`` /
    ``friction_viscous`` / ... kwargs. zero_franky accepts it as a plain dict and
    converts it server-side. The same payload applies to both the joint and the
    cartesian tracker. Returns ``{}`` when disabled, so callers can splat it.

    ``velocity_epsilon`` is the joint speed at which the ``tanh`` Coulomb term
    saturates. Joint velocities scale ~linearly with replay ``speed``, so the
    configured ``velocity_epsilon`` (interpreted at speed=1.0) is scaled by
    ``speed`` to keep the compensator equally effective across the speed sweep.
    Without this, a slow replay sits inside the smoothing band and friction is
    under-compensated. ``speed`` defaults to 1.0 (no scaling) for non-replay
    callers.
    """
    friction_cfg = cfg.get("joint_friction", {})
    if not friction_cfg or not friction_cfg.get("enabled", False):
        return {}

    coulomb = [float(v) for v in friction_cfg.coulomb]
    viscous = [float(v) for v in friction_cfg.viscous]
    max_torque_cfg = friction_cfg.get("max_torque")
    max_torque = (
        [float(v) for v in max_torque_cfg]
        if max_torque_cfg is not None
        else [1.0] * len(coulomb)
    )
    velocity_epsilon = friction_cfg.get("velocity_epsilon")
    velocity_epsilon = 0.03 if velocity_epsilon is None else float(velocity_epsilon)
    velocity_epsilon = max(velocity_epsilon * float(speed), 1e-6)  # franky requires > 0
    return {
        "friction": {
            "coulomb": coulomb,
            "viscous": viscous,
            "max_torque": max_torque,
            "velocity_epsilon": velocity_epsilon,
        }
    }


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
