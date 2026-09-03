"""Inverse kinematics for following a Cartesian path with the 7-DOF arm.

Turns tool poses into joint configurations, needing nothing but the poses. Use
`CartesianIK.solve` for one pose at a time — streaming a policy's waypoints to a
joint impedance tracker, say — and `CartesianIK.solve_trajectory` to convert a
whole recorded path up front.
"""

import logging
from dataclasses import dataclass

import numpy as np
from franky import Affine
from franky.kinematics import (
    IKOptions,
    RedundancyParameter,
    forward_kinematics,
    inverse_kinematics,
    jacobian as analytic_jacobian,
)

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    fk_f_t_ee,
)
from clear_franka.geometry import pack_Rp

logger = logging.getLogger(__name__)


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The twist (3 translation, 3 rotation) taking pose `current` onto `target`."""
    error = np.empty(6, dtype=float)
    error[:3] = target[:3, 3] - current[:3, 3]
    delta = target[:3, :3] @ current[:3, :3].T
    axis = np.array([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ]) / 2.0
    sin = np.linalg.norm(axis)
    error[3:] = axis if sin < 1e-9 else axis * (np.arctan2(sin, (np.trace(delta) - 1.0) / 2.0) / sin)
    return error


def as_matrix(pose) -> np.ndarray:
    """A 4x4 homogeneous matrix from an `Affine` or anything array-like."""
    return np.asarray(pose.matrix if isinstance(pose, Affine) else pose, dtype=float).reshape(4, 4)


def ik_frame_from_episode(episode: dict) -> tuple[np.ndarray | None, Affine | None]:
    """The seed configuration and tool frame an episode implies, if it has joints.

    IK needs neither — `CartesianIK` falls back for both — but an episode that
    does carry joint samples pins them down exactly, so it is worth reading the
    first sample for. The flange pose `forward_kinematics` derives from that
    configuration and the tool pose recorded at the same instant differ by
    precisely the flange-to-TCP offset the poses are expressed in, whichever that
    is: the controller's F_T_EE for a raw recording, or the URDF's for one whose
    Cartesian fields preprocessing recomputed by forward kinematics.

    Returns (None, None) when the episode has no usable joint samples.
    """
    joint_pos = episode.get("joint_pos")
    ee_pos, ee_rot = episode.get("ee_pos"), episode.get("ee_rot")
    if joint_pos is None or ee_pos is None or ee_rot is None:
        return None, None
    if not (
        np.all(np.isfinite(joint_pos[0]))
        and np.all(np.isfinite(ee_pos[0]))
        and np.all(np.isfinite(ee_rot[0]))
    ):
        return None, None
    q_seed = np.asarray(joint_pos[0], dtype=float)
    return q_seed, forward_kinematics(q_seed).inverse * Affine(pack_Rp(ee_rot[0], ee_pos[0]))


@dataclass(frozen=True)
class IKSolution:
    """One solve's outcome: the configuration, and how well it did.

    `reached` is the caller's cue in a control loop, where a waypoint the arm
    cannot quite make is a reason to hold or replan rather than to raise.
    """

    joint_pos: np.ndarray
    position_error: float
    rotation_error: float
    joint_motion: float
    reached: bool

    def __str__(self) -> str:
        return (
            f"{self.position_error * 1000:.3f} mm / {np.degrees(self.rotation_error):.3f}deg "
            f"from the pose, {self.joint_motion:.3f} rad from the seed"
        )


class CartesianIK:
    """Solves tool poses into joint configurations in one fixed tool frame.

    `f_t_ee` is the flange-to-TCP offset the poses are expressed in. It defaults
    to the URDF's, which is the frame `clear_franka.franka.fk_ee_poses` computes
    tool poses in and therefore the frame a preprocessed episode's `ee_pos` /
    `ee_rot` are in; `ik_frame_from_episode` recovers it from a raw recording.
    """

    def __init__(
        self,
        *,
        f_t_ee: np.ndarray | Affine | None = None,
        joint_limit_margin: float = 0.02,
        position_tolerance: float = 1e-3,
        rotation_tolerance: float = 1e-2,
        limit_activation: float = 0.15,
        limit_gain: float = 0.02,
        max_iterations: int = 40,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
    ):
        self.f_t_ee = Affine(fk_f_t_ee()) if f_t_ee is None else Affine(as_matrix(f_t_ee))
        self.lower = np.asarray(lower_joint_limits, dtype=float) + joint_limit_margin
        self.upper = np.asarray(upper_joint_limits, dtype=float) - joint_limit_margin
        self.joint_limit_margin = joint_limit_margin
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.limit_activation = limit_activation
        self.limit_gain = limit_gain
        self.max_iterations = max_iterations

    def forward(self, joint_pos: np.ndarray) -> np.ndarray:
        """The tool pose at `joint_pos`, as a 4x4 matrix in this solver's frame."""
        return forward_kinematics(joint_pos, f_t_ee=self.f_t_ee).matrix

    def jacobian(self, joint_pos: np.ndarray) -> np.ndarray:
        """The 6x7 tool Jacobian at `joint_pos`: linear rows first, in the base frame.

        Same convention `pose_error` returns a twist in, so the two compose
        directly. Nominal Franka geometry, like `forward_kinematics`, so neither
        accounts for per-robot calibration.
        """
        return analytic_jacobian(joint_pos, f_t_ee=self.f_t_ee)

    def seed_configuration(self, target) -> np.ndarray | None:
        """A start configuration for `target`, chosen for room rather than closeness.

        For when there is no previous configuration to continue from. The analytic
        solver is the right tool here and only here: it enumerates whole solution
        branches, so sweeping the redundancy (as the arm-plane, or swivel, angle)
        across its full range surveys every posture that reaches the pose, and the
        one furthest inside the joint limits has the most room to absorb wherever
        the path goes next. Returns None if the pose is out of reach entirely.
        """
        options = IKOptions(joint_limits=(self.lower, self.upper))
        pose = Affine(as_matrix(target))
        best, best_clearance = None, -np.inf
        for swivel in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            for q in inverse_kinematics(
                pose,
                float(swivel),
                parameter=RedundancyParameter.Swivel,
                f_t_ee=self.f_t_ee,
                options=options,
            ):
                q = np.asarray(q, dtype=float)
                clearance = float(np.min(np.minimum(q - self.lower, self.upper - q)))
                if clearance > best_clearance:
                    best, best_clearance = q, clearance
        return best

    def _limit_repulsion(self, joint_pos: np.ndarray) -> np.ndarray:
        """A nudge away from any joint limit `joint_pos` has come within reach of.

        Without it the solve happily converges onto a limit and stays pinned
        there — every later pose then has one fewer degree of freedom to work
        with, and the arm eventually cannot follow the path at all. This is the
        same job the impedance controller's joint-limit soft-stop does at runtime.
        """
        push = np.zeros(7, dtype=float)
        if self.limit_activation <= 0.0 or self.limit_gain <= 0.0:
            return push
        to_lower, to_upper = joint_pos - self.lower, self.upper - joint_pos
        near = to_lower < self.limit_activation
        push[near] += self.limit_gain * (self.limit_activation - to_lower[near]) / self.limit_activation
        near = to_upper < self.limit_activation
        push[near] -= self.limit_gain * (self.limit_activation - to_upper[near]) / self.limit_activation
        return push

    def solve(self, target, seed: np.ndarray) -> IKSolution:
        """The configuration nearest `seed` that reaches `target`.

        Never raises for an unreachable pose: it returns its closest attempt with
        `reached` false and the residual filled in, so a control loop can hold or
        replan on its own terms. `solve_trajectory` is the one that gives up.
        """
        target = as_matrix(target)
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        # Bias the start away from any limit the arm has drifted up against. Doing
        # it here rather than inside the iteration keeps the solve itself pure: a
        # null-space push applied during it leaks through the damped projector and
        # fights convergence, whereas a nudged starting point simply lands the
        # solve on a roomier configuration that reaches the same pose.
        q = np.clip(seed + self._limit_repulsion(seed), self.lower, self.upper)

        error = pose_error(self.forward(q), target)
        damping = 1e-4
        for _ in range(self.max_iterations):
            if np.linalg.norm(error) < 1e-9:
                break
            jacobian = self.jacobian(q)
            jjt = jacobian @ jacobian.T
            for _attempt in range(10):
                step = jacobian.T @ np.linalg.solve(jjt + damping**2 * np.eye(6), error)
                candidate = np.clip(q + step, self.lower, self.upper)
                residual = pose_error(self.forward(candidate), target)
                if np.linalg.norm(residual) < np.linalg.norm(error):
                    q, error = candidate, residual
                    damping = max(damping * 0.5, 1e-6)
                    break
                damping *= 4.0
            else:
                break  # no damping value made progress; this is as close as it gets

        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        return IKSolution(
            joint_pos=q,
            position_error=position_error,
            rotation_error=rotation_error,
            joint_motion=float(np.max(np.abs(q - seed))),
            reached=(
                position_error <= self.position_tolerance
                and rotation_error <= self.rotation_tolerance
            ),
        )

    def solve_trajectory(
        self,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        *,
        q_seed: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        max_distance: float | None = 0.2,
    ) -> np.ndarray:
        """Convert a whole Cartesian path into a joint trajectory, (N, 7).

        Each pose is solved from the previous solution, so the result is
        continuous. `q_seed` is only the posture to start from — the caller is
        expected to pre-position the arm to row 0 — and falls back to
        `seed_configuration` for the first pose.

        Raises `RuntimeError` rather than returning a trajectory that cannot be
        executed: a pose missed by more than the tolerances, or reached only by
        moving a joint further than `max_distance` in one step. Both mean the same
        thing in practice — from where the arm now is, this path cannot be
        followed any further, usually a singularity or a joint pinned against a
        limit with the path still demanding more.

        Use `solve_trajectory_prefix` instead from a live control loop, where the
        executable part of the path is more useful than an exception.
        """
        joint_pos, reason = self.solve_trajectory_prefix(
            ee_pos, ee_rot, q_seed=q_seed, timestamps=timestamps,
            max_distance=max_distance,
        )
        if reason is not None:
            raise RuntimeError(reason)
        return joint_pos

    def solve_trajectory_prefix(
        self,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        *,
        q_seed: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        max_distance: float | None = 0.2,
    ) -> tuple[np.ndarray, str | None]:
        """As much of a Cartesian path as the arm can actually follow.

        Returns `(joint_pos, reason)`: an (M, 7) joint trajectory for the leading
        M poses that solved, and `None` when M == N. Otherwise `reason` says why
        pose M could not be followed, and M may be 0.

        This is the control-loop counterpart to `solve_trajectory`. A caller
        mid-rollout cannot act on an exception — the arm is already moving — but
        it can execute the prefix and replan from wherever that leaves it.
        """
        poses = pack_Rp(ee_rot, ee_pos).reshape(len(ee_pos), 4, 4)

        def where(i: int) -> str:
            at = "" if timestamps is None else f" (t={timestamps[i]:.3f}s)"
            return f"IK failed at step {i}/{len(poses)}{at}: "

        if q_seed is None:
            q_prev = self.seed_configuration(poses[0])
            if q_prev is None:
                return np.empty((0, 7), dtype=float), (
                    where(0) + "the pose is out of reach in every configuration "
                    "within the joint limits."
                )
        else:
            q_prev = np.asarray(q_seed, dtype=float)

        joint_pos = np.empty((len(poses), 7), dtype=float)
        worst = None
        for i, target in enumerate(poses):
            solution = self.solve(target, q_prev)
            if not solution.reached:
                return joint_pos[:i].copy(), (
                    where(i) + f"closest reachable configuration still misses the pose by "
                    f"{solution.position_error * 1000:.2f} mm / "
                    f"{np.degrees(solution.rotation_error):.2f}deg, outside the "
                    f"{self.position_tolerance * 1000:.2f} mm / "
                    f"{np.degrees(self.rotation_tolerance):.2f}deg tolerance. The arm cannot "
                    f"follow the path further from here — check for a joint against its limit "
                    f"(joint_limit_margin={self.joint_limit_margin} rad)."
                )
            # Not at the first pose: `max_distance` bounds motion between poses the
            # arm will execute, and nothing has been executed yet. The seed only
            # chooses a starting posture, so it is free to sit far from the path.
            if i > 0 and max_distance is not None and solution.joint_motion > max_distance:
                moved = np.abs(solution.joint_pos - q_prev)
                return joint_pos[:i].copy(), (
                    where(i) + f"reaching it means moving j{int(np.argmax(moved)) + 1} by "
                    f"{solution.joint_motion:.3f} rad in one step, more than "
                    f"max_distance={max_distance} rad. The trajectory either crosses a "
                    f"singularity here or is sampled too coarsely to follow smoothly."
                )
            joint_pos[i] = solution.joint_pos
            q_prev = solution.joint_pos
            if worst is None or solution.position_error > worst.position_error:
                worst = solution

        if worst is not None and (worst.position_error > 1e-6 or worst.rotation_error > 1e-6):
            logger.info("ik: worst-case pose residual %s", worst)
        return joint_pos, None
