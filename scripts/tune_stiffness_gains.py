"""Interactively tune Franka impedance gains in a viser UI.

Runs a viser server whose GUI lets you:

  * pick the control mode — Cartesian impedance or per-joint impedance;
  * slide the stiffness / damping gains live (per-axis diagonal for Cartesian,
    per-joint diagonal for joint), the nullspace posture + manipulability gains
    (Cartesian only), the hybrid Cartesian gain shaping layered on top of the
    joint-space stiffness (joint only), and the friction-compensation params;
  * play a handful of built-in "reference motions" (circles, lines, squares,
    orientation wiggles, Lissajous curves, joint sinusoids) under the current gains;
  * read back the resulting tracking error (position/rotation for Cartesian,
    per-joint for joint) and see the planned-vs-actual path in the 3D scene.

Stiffness, damping, nullspace and Cartesian-shaping gains apply live while a session runs.
Friction and joint-limit safety params are baked in at session start, so changing them (or the
control mode, or whether Cartesian shaping exists at all) rebuilds the controller — use the
"Restart controller" button, or just switch mode. The robot holds its current pose whenever no
reference motion is playing.

Launch (server + robot must be reachable, same as teleop/replay):

    uv run python scripts/tune_stiffness_gains.py

Config overrides work like the other scripts, e.g.

    uv run python scripts/tune_stiffness_gains.py visualization.port=8085

Press Ctrl-C in the terminal to stop; the controller is torn down cleanly.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import hydra
import numpy as np
from omegaconf import DictConfig

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    stop_tracker_motion,
)
from threed_mouse.geometry import pack_Rp, so3_exp


# ── geometry / motion math (pure, unit-testable) ────────────────────────────────

_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}

# Integer frequency ratios (and offsetting phases) give a closed, non-repeating-looking
# curve. Ported from the Lissajous tracking-error benchmark in franky/examples.
_LISSAJOUS_FREQ_RATIOS = np.array([1.0, 2.0, 3.0])
_LISSAJOUS_PHASES = np.array([0.0, math.pi / 2.0, math.pi / 4.0])
_LISSAJOUS_AXIS_RATIOS = np.array([1.0, 1.0, 0.5])  # relative amplitude per axis (x, y, z)

CARTESIAN_MOTIONS = ("Hold", "Circle (XY)", "Line", "Square (XY)", "Orientation wiggle", "Lissajous")
JOINT_MOTIONS = ("Hold", "Single joint sine", "All joints sine")

# Diagonal ordering of a 6-DOF Cartesian stiffness, shared by the Cartesian tab and
# the joint tab's gain shaping.
_CART_AXIS_LABELS = ("tx", "ty", "tz", "rx", "ry", "rz")
_CART_AXIS_STIFFNESS_MAX = (3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0)
_CART_AXIS_DAMPING_MAX = (200.0, 200.0, 200.0, 50.0, 50.0, 50.0)

# Conventional Franka ready pose: centered and comfortably away from joint limits.
RESET_JOINT_CONFIG = np.array(
    [0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0]
)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(theta)) * theta


def rotation_error_deg(measured_R: np.ndarray, target_R: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    R_err = np.asarray(measured_R, dtype=float) @ np.asarray(target_R, dtype=float).T
    cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


@dataclass
class MotionParams:
    period_s: float = 8.0
    radius_m: float = 0.05
    amplitude_m: float = 0.05
    wiggle_deg: float = 15.0
    axis: str = "x"
    joint_index: int = 4  # 1-based
    joint_amplitude_rad: float = 0.2


def cartesian_pose(motion: str, t: float, pos0: np.ndarray, rot0: np.ndarray, p: MotionParams):
    """Return (position(3), rotation(3x3)) for a Cartesian reference at time t.

    Every motion starts at (pos0, rot0) at t=0 so engaging it never jumps.
    """
    pos0 = np.asarray(pos0, dtype=float)
    rot0 = np.asarray(rot0, dtype=float)
    w = 2.0 * math.pi / max(p.period_s, 1e-3)
    if motion == "Hold":
        return pos0.copy(), rot0.copy()
    if motion == "Circle (XY)":
        center = pos0 - p.radius_m * _AXES["x"]
        theta = w * t
        pos = center + p.radius_m * np.array([math.cos(theta), math.sin(theta), 0.0])
        return pos, rot0.copy()
    if motion == "Line":
        u = _AXES[p.axis]
        return pos0 + u * (p.amplitude_m * math.sin(w * t)), rot0.copy()
    if motion == "Square (XY)":
        side = p.amplitude_m
        corners = [
            pos0,
            pos0 + np.array([side, 0.0, 0.0]),
            pos0 + np.array([side, side, 0.0]),
            pos0 + np.array([0.0, side, 0.0]),
        ]
        frac = (t / max(p.period_s, 1e-3)) % 1.0
        seg = frac * 4.0
        i = int(seg) % 4
        local = seg - int(seg)
        a, b = corners[i], corners[(i + 1) % 4]
        return a + (b - a) * local, rot0.copy()
    if motion == "Orientation wiggle":
        u = _AXES[p.axis]
        ang = math.radians(p.wiggle_deg) * math.sin(w * t)
        return pos0.copy(), so3_exp(u * ang) @ rot0
    if motion == "Lissajous":
        amp = p.amplitude_m * _LISSAJOUS_AXIS_RATIOS
        phase = w * _LISSAJOUS_FREQ_RATIOS * t + _LISSAJOUS_PHASES
        offset0 = amp * np.sin(_LISSAJOUS_PHASES)  # curve value at t=0, so pos0 isn't a jump
        return pos0 - offset0 + amp * np.sin(phase), rot0.copy()
    raise ValueError(f"Unknown Cartesian motion: {motion!r}")


def joint_reference(motion: str, t: float, q0: np.ndarray, p: MotionParams) -> np.ndarray:
    """Return the joint reference (7,) for a joint-space motion at time t."""
    q0 = np.asarray(q0, dtype=float)
    w = 2.0 * math.pi / max(p.period_s, 1e-3)
    if motion == "Hold":
        return q0.copy()
    if motion == "Single joint sine":
        q = q0.copy()
        j = int(np.clip(p.joint_index - 1, 0, len(q) - 1))
        q[j] += p.joint_amplitude_rad * math.sin(w * t)
        return q
    if motion == "All joints sine":
        # Alternate direction per joint (not phase) so the motion starts at q0.
        direction = np.where(np.arange(len(q0)) % 2 == 0, 1.0, -1.0)
        return q0 + p.joint_amplitude_rad * direction * math.sin(w * t)
    raise ValueError(f"Unknown joint motion: {motion!r}")


def cartesian_twist(motion: str, t: float, pos0, rot0, p: MotionParams, h: float = 1e-3):
    """Central-difference (linear, angular) velocity feedforward for a motion."""
    p2, r2 = cartesian_pose(motion, t + h, pos0, rot0, p)
    p1, r1 = cartesian_pose(motion, t - h, pos0, rot0, p)
    linear = (p2 - p1) / (2.0 * h)
    angular = so3_log(r2 @ r1.T) / (2.0 * h)
    return linear, angular


def joint_velocity(motion: str, t: float, q0, p: MotionParams, h: float = 1e-3) -> np.ndarray:
    return (joint_reference(motion, t + h, q0, p) - joint_reference(motion, t - h, q0, p)) / (2.0 * h)


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "rms": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "max": float(np.max(np.abs(values))),
    }


def cartesian_error_metrics(target_pos, measured_pos, target_rot, measured_rot) -> dict:
    target_pos = np.asarray(target_pos, dtype=float)
    measured_pos = np.asarray(measured_pos, dtype=float)
    pos_err_mm = np.linalg.norm(target_pos - measured_pos, axis=1) * 1000.0
    rot_err_deg = np.array(
        [rotation_error_deg(measured_rot[i], target_rot[i]) for i in range(len(target_rot))]
    )
    return {
        "n": len(pos_err_mm),
        "position_mm": _stats(pos_err_mm),
        "rotation_deg": _stats(rot_err_deg),
    }


def joint_error_metrics(target_q, measured_q) -> dict:
    target_q = np.asarray(target_q, dtype=float)
    measured_q = np.asarray(measured_q, dtype=float)
    err_deg = np.degrees(target_q - measured_q)  # (N, 7)
    per_joint = [_stats(err_deg[:, j]) for j in range(err_deg.shape[1])]
    return {
        "n": len(err_deg),
        "per_joint_deg": per_joint,
        "overall_deg": _stats(err_deg.reshape(-1)),
    }


def critical_damping(stiffness: np.ndarray) -> np.ndarray:
    """2*sqrt(k) critical damping for unit inertia (per DOF)."""
    return 2.0 * np.sqrt(np.clip(np.asarray(stiffness, dtype=float), 0.0, None))


class CartesianImpedanceGains:
    """Wire stand-in for franky's gains type, carrying a full 6x6 stiffness/damping.

    zero_franky encodes gains by type *name* and rebuilds the real object next to the
    robot, so anisotropic gains can be sent from a client whose own franky build predates
    the matrix-valued `CartesianImpedanceGains` (older ones only expose the isotropic
    translational/rotational scalars, which would silently drop the anisotropy).
    """

    def __init__(self, stiffness, damping=None):
        self.stiffness = stiffness
        self.damping = damping


def diagonal_cartesian_gains(k6, d6=None):
    """Gains with an anisotropic (diagonal) 6-DOF stiffness.

    A `d6` of None leaves damping unpinned, so the controller keeps it critical against
    the stiffness it is currently interpolating toward rather than freezing today's value.
    """
    stiffness = np.diag(np.asarray(k6, dtype=float))
    damping = None if d6 is None else np.diag(np.asarray(d6, dtype=float))
    return CartesianImpedanceGains(stiffness, damping)


def resolve_axis_damping(k6, d6) -> np.ndarray | None:
    """Per-axis damping with zeros filled in as critical, or None when nothing is pinned.

    Cartesian damping is all-or-nothing across the 6 axes, so one pinned axis forces the
    rest to be named too -- each falls back to critical damping for its own stiffness.
    """
    d6 = np.asarray(d6, dtype=float)
    if not np.any(d6 > 0):
        return None
    return np.where(d6 > 0, d6, critical_damping(k6))


# ── shared state between the viser GUI thread and the control thread ─────────────

@dataclass
class TuningState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    restart_requested: bool = False
    move_home_requested: bool = False
    play_requested: bool = False
    stop_requested: bool = False
    quit_requested: bool = False


# ── control loop ────────────────────────────────────────────────────────────────

class GainTuner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.state = TuningState()

        tc = cfg.get("teleop", {})
        rc = cfg.get("replay", {})
        dc = cfg.get("demonstrate", {})
        self.reset_joint_config = RESET_JOINT_CONFIG.copy()
        self.default_trans_k = float(tc.get("translational_stiffness", 100.0))
        self.default_rot_k = float(tc.get("rotational_stiffness", 5.0))
        self.default_posture_k = float(tc.get("nullspace_stiffness", 2.0))
        joint_k = np.asarray(rc.get("joint_stiffness", [320, 320, 320, 320, 120, 120, 30]), dtype=float)
        if joint_k.shape != (7,):
            joint_k = np.full(7, 200.0)
        self.default_joint_k = joint_k

        fric = dc.get("joint_friction", {})
        self.default_friction_enabled = bool(fric.get("enabled", True))
        self.default_coulomb = np.asarray(fric.get("coulomb", [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2]), dtype=float)
        self.default_viscous = np.asarray(fric.get("viscous", [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05]), dtype=float)
        self.default_velocity_epsilon = float(fric.get("velocity_epsilon", 0.03))

        self.visualizer = None
        self.robot = None
        self.session = None
        self._active_kind = None
        self._hybrid_active = False  # whether the live joint session carries Cartesian shaping
        self._applied_gains = None  # snapshot of last-applied live gains

        # playback bookkeeping
        self._playing = False
        self._play_start = 0.0
        self._play_duration = 0.0
        self._play_motion = "Hold"
        self._play_params = MotionParams()
        self._origin_pos = None
        self._origin_rot = None
        self._origin_q = None
        self._rec_target_pos: list[np.ndarray] = []
        self._rec_meas_pos: list[np.ndarray] = []
        self._rec_target_rot: list[np.ndarray] = []
        self._rec_meas_rot: list[np.ndarray] = []
        self._rec_target_q: list[np.ndarray] = []
        self._rec_meas_q: list[np.ndarray] = []
        self._path_handles = []

        self.gui = None  # populated in _build_gui

    # ---- friction / session construction ----

    def _friction_kwargs(self) -> dict:
        g = self.gui
        if not g["friction_enabled"].value:
            return {}
        return {
            "friction": {
                "coulomb": [float(h.value) for h in g["coulomb"]],
                "viscous": [float(h.value) for h in g["viscous"]],
                "max_torque": [1.0] * 7,
                "velocity_epsilon": max(float(g["velocity_epsilon"].value), 1e-6),
            }
        }

    def _joint_safety_kwargs(self) -> dict:
        g = self.gui
        if not g["joint_safety_enabled"].value:
            return {}
        return {
            "lower_joint_limits": DEFAULT_LOWER_JOINT_LIMITS,
            "upper_joint_limits": DEFAULT_UPPER_JOINT_LIMITS,
            "joint_limit_activation_distance": float(g["joint_limit_activation_distance"].value),
            "joint_limit_stiffness": float(g["joint_limit_stiffness"].value),
            "joint_limit_damping": float(g["joint_limit_damping"].value),
            "joint_limit_max_torque": float(g["joint_limit_max_torque"].value),
        }

    def _build_cartesian_session(self, friction: dict):
        from franky import ManipulabilityTask, PostureTask

        # Both tasks are seeded at construction (even with a zero gain) so that later
        # live updates via set_nullspace_gains have a task to act on -- the posture
        # target and the task set itself are otherwise fixed for the session's lifetime.
        posture_task = PostureTask(
            target=self.reset_joint_config.tolist(), stiffness=float(self.gui["posture_k"].value)
        )
        manipulability_task = ManipulabilityTask(gain=float(self.gui["manip_k"].value))
        return self.robot.start_cartesian_impedance_tracker(
            period=0.001,
            translational_stiffness=float(self.gui["trans_k"].value),
            rotational_stiffness=float(self.gui["rot_k"].value),
            posture_task=posture_task,
            manipulability_task=manipulability_task,
            **self._joint_safety_kwargs(),
            **friction,
        )

    def _shaping_kwargs(self) -> dict:
        """Session kwargs enabling the hybrid Cartesian gain shaping, or {} for none.

        The hybrid path exists only when the motion is built with a `cartesian_stiffness`,
        and is fixed for the motion's lifetime -- only the gain *values* retune live.
        """
        if not self.gui["shaping_enabled"].value:
            return {}
        k6, d6 = self._cartesian_gain_vectors(self.gui["shaping_gain_group"])
        kwargs = {"cartesian_stiffness": k6.tolist()}
        if d6 is not None:
            kwargs["cartesian_damping"] = d6.tolist()
        return kwargs

    def _build_joint_session(self, friction: dict):
        stiffness = [float(h.value) for h in self.gui["joint_k"]]
        shaping = self._shaping_kwargs()
        base = dict(period=0.001, stiffness=stiffness, **self._joint_safety_kwargs(), **friction)
        try:
            session = self.robot.start_joint_impedance_tracker(**base, **shaping)
        except Exception as exc:  # noqa: BLE001
            if not shaping:
                raise
            # Shaping needs a server-side franky carrying the hybrid joint impedance path;
            # without it the motion never gets built, so fall back rather than leaving the
            # tuner with no controller at all.
            print(f"  [tuner] Cartesian gain shaping rejected: {exc}")
            self.gui["results"].content = (
                "**Cartesian gain shaping unavailable** — started plain joint impedance instead.\n\n"
                f"`{exc}`\n"
            )
            session = self.robot.start_joint_impedance_tracker(**base)
            shaping = {}
        self._hybrid_active = bool(shaping)
        return session

    def _rebuild_session(self, kind: str):
        self._playing = False
        self._hybrid_active = False
        self._clear_paths()
        if self.session is not None:
            try:
                stop_tracker_motion(self.robot, self.session, join_timeout=1.0, idle_timeout_s=2.0)
            except Exception as exc:  # noqa: BLE001
                print(f"  [tuner] stopping old session failed: {exc}")
            self.session = None

        self.robot.recover_from_errors()
        friction = self._friction_kwargs()
        print(f"  [tuner] starting {kind} session (friction {'on' if friction else 'off'})")
        if kind == "cartesian":
            self.session = self._build_cartesian_session(friction)
        else:
            self.session = self._build_joint_session(friction)
            if self._hybrid_active:
                print("  [tuner] Cartesian gain shaping on")
        self._active_kind = kind
        self._applied_gains = None  # force a live-gain apply next tick

    # ---- live gain application ----

    @staticmethod
    def _cartesian_gain_vectors(group) -> tuple[np.ndarray, np.ndarray | None]:
        """One group of Cartesian sliders as a (stiffness(6), damping(6)-or-None) diagonal.

        The isotropic sliders are just the diagonal with the three translational and the
        three rotational axes tied together, so both modes reach the controller the same
        way: as a full 6-DOF diagonal, which is the only form that can carry anisotropy.
        """
        per_axis, axis_k, axis_d, trans_k, rot_k, trans_d, rot_d = group
        if per_axis.value:
            k6 = np.array([float(h.value) for h in axis_k], dtype=float)
            d6 = np.array([float(h.value) for h in axis_d], dtype=float)
        else:
            k6 = np.array([float(trans_k.value)] * 3 + [float(rot_k.value)] * 3)
            d6 = np.array([float(trans_d.value)] * 3 + [float(rot_d.value)] * 3)
        return k6, resolve_axis_damping(k6, d6)

    def _cartesian_gain_key(self, group) -> tuple:
        """Change-detection key for one group of Cartesian stiffness/damping sliders."""
        k6, d6 = self._cartesian_gain_vectors(group)
        return (
            tuple(np.round(k6, 3)),
            None if d6 is None else tuple(np.round(d6, 3)),
        )

    def _gain_snapshot(self):
        g = self.gui
        if self._active_kind == "cartesian":
            nullspace = (
                round(float(g["posture_k"].value), 3),
                round(float(g["posture_d"].value), 3),
                round(float(g["manip_k"].value), 3),
                round(float(g["manip_d"].value), 3),
            )
            return (self._cartesian_gain_key(g["cart_gain_group"]), nullspace)
        return (
            "joint",
            tuple(round(float(h.value), 3) for h in g["joint_k"]),
            tuple(round(float(h.value), 3) for h in g["joint_d"]),
            self._cartesian_gain_key(g["shaping_gain_group"]) if self._hybrid_active else None,
        )

    def _apply_gains(self):
        g = self.gui
        if self._active_kind == "cartesian":
            self.session.set_gains(diagonal_cartesian_gains(*self._cartesian_gain_vectors(g["cart_gain_group"])))
            posture_d = float(g["posture_d"].value)
            self.session.set_nullspace_gains(
                posture_stiffness=float(g["posture_k"].value),
                posture_damping=posture_d if posture_d > 0 else None,
                manipulability_gain=float(g["manip_k"].value),
                manipulability_damping=float(g["manip_d"].value),
            )
        else:
            k7 = np.array([float(h.value) for h in g["joint_k"]], dtype=float)
            d7 = np.array([float(h.value) for h in g["joint_d"]], dtype=float)
            auto = critical_damping(k7)
            d7 = np.where(d7 > 0, d7, auto)
            self.session.set_gains(stiffness=k7, damping=d7)
            if self._hybrid_active:
                self.session.set_cartesian_gains(
                    diagonal_cartesian_gains(*self._cartesian_gain_vectors(g["shaping_gain_group"]))
                )

    # ---- playback ----

    def _read_params(self) -> MotionParams:
        g = self.gui
        return MotionParams(
            period_s=float(g["period_s"].value),
            radius_m=float(g["radius_m"].value),
            amplitude_m=float(g["amplitude_m"].value),
            wiggle_deg=float(g["wiggle_deg"].value),
            axis=str(g["axis"].value),
            joint_index=int(g["joint_index"].value),
            joint_amplitude_rad=float(g["joint_amp"].value),
        )

    def _start_playback(self, teleop_state):
        self._play_motion = str(self.gui["motion"].value)
        self._play_params = self._read_params()
        self._play_duration = float(self.gui["duration_s"].value)
        self._play_start = time.monotonic()
        self._rec_target_pos.clear()
        self._rec_meas_pos.clear()
        self._rec_target_rot.clear()
        self._rec_meas_rot.clear()
        self._rec_target_q.clear()
        self._rec_meas_q.clear()
        pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
        self._origin_pos = pose[:3, 3].copy()
        self._origin_rot = pose[:3, :3].copy()
        self._origin_q = np.asarray(teleop_state["q"], dtype=float).copy()
        self._playing = True
        self.gui["results"].content = f"Playing **{self._play_motion}** for {self._play_duration:.0f}s…"
        print(f"  [tuner] play {self._active_kind}/{self._play_motion} for {self._play_duration:.0f}s")

    def _step_playback(self, teleop_state) -> bool:
        """Advance one playback tick. Returns True while still playing."""
        t = time.monotonic() - self._play_start
        if t >= self._play_duration:
            return False

        if self._active_kind == "cartesian":
            from franky import Affine, Twist

            target_pos, target_rot = cartesian_pose(
                self._play_motion, t, self._origin_pos, self._origin_rot, self._play_params
            )
            lin, ang = cartesian_twist(
                self._play_motion, t, self._origin_pos, self._origin_rot, self._play_params
            )
            try:
                self.session.set_target(
                    Affine(pack_Rp(target_rot, target_pos)), Twist(lin, ang)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [tuner] set_target failed: {exc}")
            meas = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
            self._rec_target_pos.append(target_pos)
            self._rec_meas_pos.append(meas[:3, 3].copy())
            self._rec_target_rot.append(target_rot)
            self._rec_meas_rot.append(meas[:3, :3].copy())
        else:
            target_q = joint_reference(self._play_motion, t, self._origin_q, self._play_params)
            dq = joint_velocity(self._play_motion, t, self._origin_q, self._play_params)
            try:
                self.session.set_target(target_q, dq=dq)
            except Exception as exc:  # noqa: BLE001
                print(f"  [tuner] set_target failed: {exc}")
            self._rec_target_q.append(target_q)
            self._rec_meas_q.append(np.asarray(teleop_state["q"], dtype=float).copy())
        return True

    def _finish_playback(self, aborted: bool):
        self._playing = False
        if self._active_kind == "cartesian" and self._rec_meas_pos:
            metrics = cartesian_error_metrics(
                self._rec_target_pos, self._rec_meas_pos, self._rec_target_rot, self._rec_meas_rot
            )
            self._draw_cartesian_paths()
            report = self._format_cartesian_report(metrics, aborted)
        elif self._active_kind == "joint" and self._rec_meas_q:
            metrics = joint_error_metrics(self._rec_target_q, self._rec_meas_q)
            report = self._format_joint_report(metrics, aborted)
        else:
            report = "_No samples recorded._"
        self.gui["results"].content = report
        print("  [tuner] " + report.replace("\n", " ").replace("**", ""))

    @staticmethod
    def _format_cartesian_report(m: dict, aborted: bool) -> str:
        p = m["position_mm"]
        r = m["rotation_deg"]
        head = "Stopped early — " if aborted else ""
        return (
            f"**{head}Cartesian tracking error** ({m['n']} samples)\n\n"
            f"| | mean | rms | max |\n|---|---|---|---|\n"
            f"| position [mm] | {p['mean']:.2f} | {p['rms']:.2f} | {p['max']:.2f} |\n"
            f"| rotation [deg] | {r['mean']:.2f} | {r['rms']:.2f} | {r['max']:.2f} |\n"
        )

    @staticmethod
    def _format_joint_report(m: dict, aborted: bool) -> str:
        head = "Stopped early — " if aborted else ""
        rows = "\n".join(
            f"| J{j + 1} | {s['mean']:.2f} | {s['rms']:.2f} | {s['max']:.2f} |"
            for j, s in enumerate(m["per_joint_deg"])
        )
        o = m["overall_deg"]
        return (
            f"**{head}Joint tracking error** ({m['n']} samples) [deg]\n\n"
            f"| joint | mean | rms | max |\n|---|---|---|---|\n{rows}\n"
            f"| **all** | {o['mean']:.2f} | {o['rms']:.2f} | {o['max']:.2f} |\n"
        )

    # ---- scene path drawing ----

    def _to_root(self, pts_base: np.ndarray) -> np.ndarray:
        T = self.visualizer.urdf_model.get_transform("fr3_link0")
        pts = np.asarray(pts_base, dtype=float)
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    def _draw_path(self, name: str, pts_base: np.ndarray, color):
        pts = self._to_root(pts_base).astype(np.float32)
        if len(pts) < 2:
            return
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = np.full((len(segments), 2, 3), color, dtype=np.uint8)
        handle = self.visualizer.server.scene.add_line_segments(
            name=name, points=segments, colors=colors, line_width=2.5
        )
        self._path_handles.append(handle)

    def _clear_paths(self):
        for handle in self._path_handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._path_handles = []

    def _draw_cartesian_paths(self):
        self._clear_paths()
        self._draw_path("/tuner/planned", np.asarray(self._rec_target_pos), (80, 160, 255))
        self._draw_path("/tuner/actual", np.asarray(self._rec_meas_pos), (60, 220, 90))

    # ---- GUI ----

    @staticmethod
    def _add_axis_sliders(gui, k_prefix: str, d_prefix: str, k_init: list[float]):
        """One hidden stiffness + damping slider per Cartesian axis, as (stiffness, damping)."""
        stiffness = [
            gui.add_slider(f"{k_prefix} {lbl}", 0.0, mx, 1.0, init, visible=False)
            for lbl, mx, init in zip(_CART_AXIS_LABELS, _CART_AXIS_STIFFNESS_MAX, k_init)
        ]
        damping = [
            gui.add_slider(f"{d_prefix} {lbl} (0=auto)", 0.0, mx, 0.5, 0.0, visible=False)
            for lbl, mx in zip(_CART_AXIS_LABELS, _CART_AXIS_DAMPING_MAX)
        ]
        return stiffness, damping

    def _build_gui(self, server):
        gui = server.gui
        handles: dict = {}

        gui.add_markdown(
            "## Franka gain tuner\n"
            "Gains apply live. Friction, joint-limit safety, and mode changes rebuild the controller."
        )
        handles["mode"] = gui.add_dropdown("Control mode", ("Cartesian", "Joint impedance"), initial_value="Cartesian")
        with gui.add_folder("Controller"):
            handles["restart"] = gui.add_button("Restart controller (apply session params)")
            handles["move_home"] = gui.add_button("Move to reset config")

        # Viser's real tab widget is client-side-only (no server->client "active tab"
        # message), so it can't be switched programmatically when the control mode
        # changes. These two top-level folders stand in for tabs instead: only the one
        # matching the current mode is visible, toggled from the mode dropdown below.
        handles["cartesian_tab"] = gui.add_folder("Cartesian impedance", visible=True)
        with handles["cartesian_tab"]:
            with gui.add_folder("Gains"):
                handles["per_axis"] = gui.add_checkbox("Per-axis diagonal stiffness", False)
                # Per-axis mode ignores trans_k/rot_k/trans_d/rot_d entirely, so only one
                # set of sliders is ever live -- hide the other to match.
                handles["trans_k"] = gui.add_slider("Translational stiffness [N/m]", 0.0, 3000.0, 5.0, self.default_trans_k)
                handles["rot_k"] = gui.add_slider("Rotational stiffness [Nm/rad]", 0.0, 300.0, 1.0, self.default_rot_k)
                handles["trans_d"] = gui.add_slider("Translational damping (0=auto)", 0.0, 200.0, 1.0, 0.0)
                handles["rot_d"] = gui.add_slider("Rotational damping (0=auto)", 0.0, 50.0, 0.5, 0.0)
                handles["cart_axis_k"], handles["cart_axis_d"] = self._add_axis_sliders(
                    gui, "K", "D", [self.default_trans_k] * 3 + [self.default_rot_k] * 3
                )
                handles["cart_gain_group"] = (
                    handles["per_axis"],
                    handles["cart_axis_k"],
                    handles["cart_axis_d"],
                    handles["trans_k"],
                    handles["rot_k"],
                    handles["trans_d"],
                    handles["rot_d"],
                )

            with gui.add_folder("Nullspace"):
                handles["posture_k"] = gui.add_slider("Posture stiffness", 0.0, 100.0, 0.5, self.default_posture_k)
                handles["posture_d"] = gui.add_slider("Posture damping (0=auto)", 0.0, 40.0, 0.5, 0.0)
                handles["manip_k"] = gui.add_slider("Manipulability gain", 0.0, 20.0, 0.1, 0.0)
                handles["manip_d"] = gui.add_slider("Manipulability damping", 0.0, 20.0, 0.1, 0.0)

        handles["joint_tab"] = gui.add_folder("Joint impedance", visible=False)
        with handles["joint_tab"]:
            with gui.add_folder("Gains"):
                handles["joint_k"] = [
                    gui.add_slider(f"J{j + 1} stiffness", 0.0, 1000.0, 5.0, float(self.default_joint_k[j]))
                    for j in range(7)
                ]
                handles["joint_d"] = [
                    gui.add_slider(f"J{j + 1} damping (0=auto)", 0.0, 100.0, 0.5, 0.0) for j in range(7)
                ]

            # Gain shaping adds a task-space impedance term (mapped through J^T) on top of
            # the joint-space stiffness, so the arm can be stiff along a Cartesian axis
            # without stiffening every joint. Whether the hybrid term exists is fixed when
            # the motion is built -- hence the rebuild on toggle -- but its gains retune live.
            with gui.add_folder("Cartesian gain shaping"):
                handles["shaping_enabled"] = gui.add_checkbox("Enabled (rebuilds controller)", False)
                handles["shaping_per_axis"] = gui.add_checkbox(
                    "Per-axis diagonal stiffness", False, visible=False
                )
                handles["shaping_trans_k"] = gui.add_slider(
                    "Translational stiffness [N/m]", 0.0, 3000.0, 5.0, self.default_trans_k, visible=False
                )
                handles["shaping_rot_k"] = gui.add_slider(
                    "Rotational stiffness [Nm/rad]", 0.0, 300.0, 1.0, self.default_rot_k, visible=False
                )
                handles["shaping_trans_d"] = gui.add_slider(
                    "Translational damping (0=auto)", 0.0, 200.0, 1.0, 0.0, visible=False
                )
                handles["shaping_rot_d"] = gui.add_slider(
                    "Rotational damping (0=auto)", 0.0, 50.0, 0.5, 0.0, visible=False
                )
                handles["shaping_axis_k"], handles["shaping_axis_d"] = self._add_axis_sliders(
                    gui, "Kc", "Dc", [self.default_trans_k] * 3 + [self.default_rot_k] * 3
                )
                handles["shaping_gain_group"] = (
                    handles["shaping_per_axis"],
                    handles["shaping_axis_k"],
                    handles["shaping_axis_d"],
                    handles["shaping_trans_k"],
                    handles["shaping_rot_k"],
                    handles["shaping_trans_d"],
                    handles["shaping_rot_d"],
                )

        with gui.add_folder("Friction compensation (restart to apply)"):
            handles["friction_enabled"] = gui.add_checkbox("Enabled", self.default_friction_enabled)
            handles["coulomb"] = [
                gui.add_slider(f"J{j + 1} coulomb", 0.0, 3.0, 0.01, float(self.default_coulomb[j]))
                for j in range(7)
            ]
            handles["viscous"] = [
                gui.add_slider(f"J{j + 1} viscous", 0.0, 1.0, 0.005, float(self.default_viscous[j]))
                for j in range(7)
            ]
            handles["velocity_epsilon"] = gui.add_slider(
                "Velocity epsilon", 0.001, 0.2, 0.001, self.default_velocity_epsilon
            )

        with gui.add_folder("Joint-limit safety (restart to apply)"):
            handles["joint_safety_enabled"] = gui.add_checkbox("Enabled", True)
            handles["joint_limit_activation_distance"] = gui.add_slider(
                "Activation distance [rad]", 0.01, 0.5, 0.01, 0.1
            )
            handles["joint_limit_stiffness"] = gui.add_slider(
                "Repulsion stiffness [Nm]", 0.0, 20.0, 0.25, 4.0
            )
            handles["joint_limit_damping"] = gui.add_slider(
                "Approach damping [Nms/rad]", 0.0, 10.0, 0.1, 1.0
            )
            handles["joint_limit_max_torque"] = gui.add_slider(
                "Maximum repulsion torque [Nm]", 0.0, 20.0, 0.25, 5.0
            )

        with gui.add_folder("Reference motion"):
            handles["motion"] = gui.add_dropdown("Motion", CARTESIAN_MOTIONS, initial_value="Hold")
            handles["duration_s"] = gui.add_slider("Duration [s]", 1.0, 60.0, 1.0, 16.0)
            handles["period_s"] = gui.add_slider("Period [s]", 1.0, 30.0, 0.5, 8.0)

            handles["cart_motion_params"] = gui.add_folder(None, visible=True)
            with handles["cart_motion_params"]:
                handles["radius_m"] = gui.add_slider("Circle radius [m]", 0.01, 0.25, 0.005, 0.05)
                handles["amplitude_m"] = gui.add_slider("Line/square amplitude [m]", 0.01, 0.3, 0.005, 0.05)
                handles["wiggle_deg"] = gui.add_slider("Orientation amplitude [deg]", 1.0, 60.0, 1.0, 15.0)
                handles["axis"] = gui.add_dropdown("Axis (line/wiggle)", ("x", "y", "z"), initial_value="x")

            handles["joint_motion_params"] = gui.add_folder(None, visible=False)
            with handles["joint_motion_params"]:
                handles["joint_index"] = gui.add_dropdown("Joint (single)", tuple(str(i) for i in range(1, 8)), initial_value="4")
                handles["joint_amp"] = gui.add_slider("Joint amplitude [rad]", 0.02, 1.0, 0.02, 0.2)

            handles["play"] = gui.add_button("▶ Play")
            handles["stop"] = gui.add_button("■ Stop")

        with gui.add_folder("Tracking results"):
            handles["results"] = gui.add_markdown("_Play a reference motion to measure tracking error._")

        # Wire buttons/mode to the shared flags (GUI thread only touches flags).
        st = self.state

        @handles["restart"].on_click
        def _(_):  # noqa: ANN001
            with st.lock:
                st.restart_requested = True

        @handles["move_home"].on_click
        def _(_):  # noqa: ANN001
            with st.lock:
                st.move_home_requested = True

        @handles["play"].on_click
        def _(_):  # noqa: ANN001
            with st.lock:
                st.play_requested = True

        @handles["stop"].on_click
        def _(_):  # noqa: ANN001
            with st.lock:
                st.stop_requested = True

        @handles["mode"].on_update
        def _(_):  # noqa: ANN001
            is_cart = handles["mode"].value == "Cartesian"
            handles["motion"].options = CARTESIAN_MOTIONS if is_cart else JOINT_MOTIONS
            handles["motion"].value = "Hold"
            handles["cartesian_tab"].visible = is_cart
            handles["joint_tab"].visible = not is_cart
            handles["cart_motion_params"].visible = is_cart
            handles["joint_motion_params"].visible = not is_cart

        def _sync_gain_visibility(group, shown: bool = True):
            """Show whichever of the group's two slider sets the per-axis toggle selects."""
            per_axis, axis_k, axis_d, trans_k, rot_k, trans_d, rot_d = group
            is_per_axis = per_axis.value
            per_axis.visible = shown
            for h in (trans_k, rot_k, trans_d, rot_d):
                h.visible = shown and not is_per_axis
            for h in (*axis_k, *axis_d):
                h.visible = shown and is_per_axis

        @handles["per_axis"].on_update
        def _(_):  # noqa: ANN001
            _sync_gain_visibility(handles["cart_gain_group"])

        def _sync_shaping_visibility():
            _sync_gain_visibility(handles["shaping_gain_group"], handles["shaping_enabled"].value)

        @handles["shaping_enabled"].on_update
        def _(_):  # noqa: ANN001
            _sync_shaping_visibility()
            # Adding or dropping the hybrid term means a new motion, so rebuild the way a
            # mode change does rather than leaving the sliders pointing at a session that
            # has no Cartesian gains to retune.
            with st.lock:
                st.restart_requested = True

        @handles["shaping_per_axis"].on_update
        def _(_):  # noqa: ANN001
            _sync_shaping_visibility()

        self.gui = handles

    # ---- main loop ----

    def run(self):
        from zero_franky import Robot, setup_zero_franky
        from franky import JointMotion, JointState

        cfg = self.cfg
        setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port)

        from clear_franka.visualization import CortadoViserVisualizer

        vc = cfg.get("visualization", {})
        self.visualizer = CortadoViserVisualizer(
            host=vc.get("host", "0.0.0.0"), port=int(vc.get("port", 8080))
        )
        self._build_gui(self.visualizer.server)

        self.robot = Robot(cfg.robot.ip)
        self.robot.recover_from_errors()

        self._rebuild_session("cartesian")

        control_hz = 50.0
        period = 1.0 / control_hz
        print("  [tuner] ready. Open the viser URL above. Ctrl-C to stop.")

        st = self.state
        try:
            while True:
                tick_start = time.monotonic()

                with st.lock:
                    restart = st.restart_requested
                    move_home = st.move_home_requested
                    play = st.play_requested
                    stop = st.stop_requested
                    st.restart_requested = False
                    st.move_home_requested = False
                    st.play_requested = False
                    st.stop_requested = False
                desired_kind = "cartesian" if self.gui["mode"].value == "Cartesian" else "joint"

                if move_home:
                    self._playing = False
                    self._clear_paths()
                    if self.session is not None:
                        try:
                            stop_tracker_motion(self.robot, self.session, join_timeout=1.0, idle_timeout_s=2.0)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [tuner] stop before home failed: {exc}")
                        self.session = None
                    print("  [tuner] moving to reset config…")
                    self.robot.recover_from_errors()
                    self.robot.move(
                        JointMotion(JointState(self.reset_joint_config.tolist()), relative_dynamics_factor=0.1)
                    )
                    self._rebuild_session(desired_kind)
                elif desired_kind != self._active_kind:
                    self._rebuild_session(desired_kind)
                elif restart:
                    self._rebuild_session(desired_kind)

                # Read state (only valid once the session callback has fired).
                try:
                    teleop_state = self.robot.get_last_teleop_state()
                except Exception:  # noqa: BLE001
                    self._sleep_to(tick_start, period)
                    continue

                self.visualizer.update(np.asarray(teleop_state["q"], dtype=float))
                self.visualizer.update_eef_frame(
                    np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                )

                # Live gains.
                snapshot = self._gain_snapshot()
                if snapshot != self._applied_gains:
                    try:
                        self._apply_gains()
                        self._applied_gains = snapshot
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [tuner] apply_gains failed: {exc}")

                # Playback.
                if play and not self._playing:
                    self._start_playback(teleop_state)
                if self._playing:
                    if stop:
                        self._finish_playback(aborted=True)
                    elif not self._step_playback(teleop_state):
                        self._finish_playback(aborted=False)

                self._sleep_to(tick_start, period)
        except KeyboardInterrupt:
            print("\n  [tuner] stopping.")
        finally:
            if self.session is not None:
                try:
                    stop_tracker_motion(self.robot, self.session, join_timeout=1.0, idle_timeout_s=2.0)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _sleep_to(tick_start: float, period: float):
        remaining = period - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)


# This script lives in scripts/, so conf/ is one level up.
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    GainTuner(cfg).run()


if __name__ == "__main__":
    main()
