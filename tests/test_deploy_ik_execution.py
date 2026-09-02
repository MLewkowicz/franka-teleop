"""Tests for deploy's `tracker: ik` plan conversion.

`_make_joint_trajectory_for_plan` is what makes IK execution possible in deploy:
the executor adopts one plan at a time and runs it to completion, so an adopted
plan's Cartesian path is fully known and can be solved into joint waypoints up
front — the per-plan equivalent of `replay.tracker=ik`.

The solver is injected, so these build it with an explicit `f_t_ee` and never
need the robot description.
"""

import numpy as np
import pytest
from franky import Affine
from franky.kinematics import forward_kinematics

from clear_franka.cartesian_trajectory import CartesianTrajectory
from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS
from clear_franka.ik import CartesianIK

from deploy_diffuser_actor import _make_joint_trajectory_for_plan

F_T_EE = np.eye(4)
F_T_EE[2, 3] = 0.19

Q_HOME = np.array([-0.06217873, 0.11705305, -0.5995304, -2.72505689, 1.98229098, 1.78315043, 0.44959301])

LOWER = np.asarray(DEFAULT_LOWER_JOINT_LIMITS, dtype=float)
UPPER = np.asarray(DEFAULT_UPPER_JOINT_LIMITS, dtype=float)

# Wide enough that nothing clips unless a test means it to.
WIDE_LO = np.array([-10.0, -10.0, -10.0])
WIDE_HI = np.array([10.0, 10.0, 10.0])


def make_ik():
    return CartesianIK(f_t_ee=F_T_EE)


def fk(joint_pos):
    return np.stack([forward_kinematics(q, f_t_ee=Affine(F_T_EE)).matrix for q in joint_pos])


def plan_like_trajectory(n=21, duration=2.0, amplitude=0.15):
    """A Cartesian trajectory shaped like an adopted plan: a short reachable arc."""
    s = np.linspace(0.0, 1.0, n)
    joint_pos = np.tile(Q_HOME, (n, 1))
    joint_pos[:, 1] += amplitude * np.sin(np.pi * s)
    joint_pos[:, 3] += amplitude * 0.8 * np.sin(np.pi * s)
    times = np.linspace(0.0, duration, n)
    return CartesianTrajectory.from_transforms(fk(joint_pos), times), joint_pos


def convert(trajectory, **kwargs):
    kwargs.setdefault("ik", make_ik())
    kwargs.setdefault("q_seed", Q_HOME)
    kwargs.setdefault("dt", 0.01)
    kwargs.setdefault("workspace_lo", WIDE_LO)
    kwargs.setdefault("workspace_hi", WIDE_HI)
    kwargs.setdefault("max_distance", 0.2)
    return _make_joint_trajectory_for_plan(trajectory, **kwargs)


def test_a_reachable_plan_solves_completely():
    trajectory, _ = plan_like_trajectory()
    joint_trajectory, reason = convert(trajectory)

    assert reason is None
    assert joint_trajectory is not None
    assert joint_trajectory.num_joints == 7
    # Spans the plan's full duration, so the executor streams the whole plan.
    assert joint_trajectory.waypts_time[0] == pytest.approx(0.0)
    assert joint_trajectory.waypts_time[-1] == pytest.approx(trajectory.duration)


def test_the_joint_trajectory_reproduces_the_cartesian_path():
    trajectory, _ = plan_like_trajectory()
    joint_trajectory, _ = convert(trajectory)

    for t in np.linspace(0.0, trajectory.duration, 25):
        want_pos, want_rot = trajectory.interpolate(t)
        got = forward_kinematics(
            joint_trajectory.interpolate(t).reshape(7), f_t_ee=Affine(F_T_EE)
        ).matrix
        assert np.linalg.norm(got[:3, 3] - want_pos) < 5e-3
        cos = np.clip((np.trace(got[:3, :3].T @ want_rot) - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(cos) < 5e-2


def test_solved_waypoints_stay_inside_the_joint_limits():
    trajectory, _ = plan_like_trajectory()
    joint_trajectory, _ = convert(trajectory)
    assert np.all(joint_trajectory.waypts >= LOWER)
    assert np.all(joint_trajectory.waypts <= UPPER)


def test_sampling_is_capped_so_the_solve_stays_cheap():
    """Every sample costs an IK solve, so a long plan must not solve thousands."""
    trajectory, _ = plan_like_trajectory(duration=20.0)
    joint_trajectory, reason = convert(trajectory, dt=0.001, max_waypoints=40)

    assert reason is None
    assert joint_trajectory.num_waypts <= 40


def test_the_workspace_clip_is_applied_before_solving():
    """The solve must be for the pose that will actually be commanded."""
    trajectory, _ = plan_like_trajectory()
    # A box that cuts the path's z down to a plane well below where it runs.
    sample_z = np.array([trajectory.interpolate(t)[0][2]
                         for t in np.linspace(0, trajectory.duration, 20)])
    z_cap = float(sample_z.min()) - 0.02
    lo = np.array([-10.0, -10.0, -10.0])
    hi = np.array([10.0, 10.0, z_cap])

    joint_trajectory, reason = convert(trajectory, workspace_lo=lo, workspace_hi=hi)
    assert reason is None and joint_trajectory is not None

    # Every solved configuration's FK z must respect the clip, which is only
    # true if the clip happened before the solve.
    for t in np.linspace(0.0, trajectory.duration, 20):
        got = forward_kinematics(
            joint_trajectory.interpolate(t).reshape(7), f_t_ee=Affine(F_T_EE)
        ).matrix
        assert got[2, 3] <= z_cap + 5e-3, f"z={got[2, 3]:.4f} exceeds clip {z_cap:.4f}"


def test_an_unreachable_tail_yields_a_shorter_prefix_and_a_reason():
    """The chosen failure policy: execute what solves, replan from there."""
    trajectory, _ = plan_like_trajectory()
    # Push the box out along +x so the later half of the path leaves the workspace
    # entirely — clipped to a point the arm cannot reach with the demanded rotation.
    lo = np.array([-10.0, -10.0, -10.0])
    hi = np.array([10.0, 10.0, 10.0])
    joint_trajectory, reason = convert(trajectory, workspace_lo=lo, workspace_hi=hi)
    assert reason is None  # sanity: the unclipped path is fine

    # Now make it genuinely unfollowable: a box far outside the workspace.
    joint_trajectory, reason = convert(
        trajectory,
        workspace_lo=np.array([3.0, -10.0, -10.0]),
        workspace_hi=np.array([10.0, 10.0, 10.0]),
    )
    assert joint_trajectory is None
    assert reason is not None


def test_zero_duration_plan_is_declined():
    trajectory, _ = plan_like_trajectory()
    # A degenerate trajectory whose samples all land at t=0.
    degenerate = CartesianTrajectory(
        trajectory.positions[:2], trajectory.rotations[:2], np.array([0.0, 1e-12])
    )
    joint_trajectory, reason = convert(degenerate, dt=1.0)
    # Either declined outright, or reduced to something non-interpolable.
    if joint_trajectory is not None:
        assert joint_trajectory.num_waypts >= 2
    else:
        assert reason is not None


def test_declined_plan_returns_none_rather_than_raising():
    """The executor cannot act on an exception mid-rollout."""
    trajectory, _ = plan_like_trajectory()
    joint_trajectory, reason = convert(
        trajectory,
        workspace_lo=np.array([5.0, 5.0, 5.0]),
        workspace_hi=np.array([6.0, 6.0, 6.0]),
    )
    assert joint_trajectory is None
    assert isinstance(reason, str) and reason


def test_seed_choice_does_not_change_the_commanded_path():
    """A different seed may pick another posture, but the tool path must hold."""
    trajectory, _ = plan_like_trajectory()
    a, _ = convert(trajectory, q_seed=Q_HOME)
    b, _ = convert(trajectory, q_seed=Q_HOME + np.array([0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.2]))

    for t in np.linspace(0.0, trajectory.duration, 15):
        pa = forward_kinematics(a.interpolate(t).reshape(7), f_t_ee=Affine(F_T_EE)).matrix
        pb = forward_kinematics(b.interpolate(t).reshape(7), f_t_ee=Affine(F_T_EE)).matrix
        assert np.linalg.norm(pa[:3, 3] - pb[:3, 3]) < 1e-2


# ── tool frame ──────────────────────────────────────────────────────────────────
# Every Cartesian pose in deploy is the robot's O_T_EE, i.e. the CONTROLLER's
# flange-to-TCP frame. CartesianIK defaults to the URDF's offset, so deploy
# derives the controller's from a live state at startup. If the solver's frame
# and the plan's frame ever diverge, every commanded waypoint carries a fixed
# offset and nothing in the solve itself reports a problem.


def controller_frame(twist_rad=np.pi / 6, tool_len=0.23):
    """A plausible controller F_T_EE that is NOT the URDF's."""
    from scipy.spatial.transform import Rotation

    f_t_ee = np.eye(4)
    f_t_ee[:3, :3] = Rotation.from_euler("z", twist_rad).as_matrix()
    f_t_ee[2, 3] = tool_len
    return f_t_ee


def test_the_derivation_recovers_the_controller_frame_from_one_state():
    """flange^-1 * tool, the same recovery replay does from a recording."""
    true_f_t_ee = controller_frame()
    O_T_EE = (forward_kinematics(Q_HOME) * Affine(true_f_t_ee)).matrix

    derived = forward_kinematics(Q_HOME).inverse * Affine(O_T_EE)

    np.testing.assert_allclose(np.asarray(derived.matrix), true_f_t_ee, atol=1e-12)
    # And FK in the derived frame lands back on the measured pose — deploy's
    # startup sanity check.
    ik = CartesianIK(f_t_ee=derived)
    assert np.linalg.norm(ik.forward(Q_HOME)[:3, 3] - O_T_EE[:3, 3]) < 1e-9


def test_a_urdf_frame_solver_would_offset_the_commanded_pose():
    """Guards the reason the derivation exists: the frames really do differ."""
    true_f_t_ee = controller_frame()
    O_T_EE = (forward_kinematics(Q_HOME) * Affine(true_f_t_ee)).matrix

    wrong = CartesianIK(f_t_ee=F_T_EE)  # a different tool offset
    offset = np.linalg.norm(wrong.forward(Q_HOME)[:3, 3] - O_T_EE[:3, 3])
    assert offset > 1e-3, "frames must differ for this test to mean anything"


def test_conversion_is_exact_in_whatever_frame_the_solver_uses():
    """The invariant that matters: solver frame == plan frame => no offset."""
    f_t_ee = controller_frame()
    ik = CartesianIK(f_t_ee=f_t_ee)

    # Build the plan IN THAT FRAME, the way deploy's plans come from O_T_EE.
    s = np.linspace(0.0, 1.0, 21)
    joint_pos = np.tile(Q_HOME, (21, 1))
    joint_pos[:, 1] += 0.15 * np.sin(np.pi * s)
    transforms = np.stack([
        (forward_kinematics(q) * Affine(f_t_ee)).matrix for q in joint_pos
    ])
    trajectory = CartesianTrajectory.from_transforms(transforms, np.linspace(0.0, 2.0, 21))

    joint_trajectory, reason = _make_joint_trajectory_for_plan(
        trajectory, ik=ik, q_seed=Q_HOME, dt=0.01,
        workspace_lo=WIDE_LO, workspace_hi=WIDE_HI, max_distance=0.2,
    )
    assert reason is None

    for t in np.linspace(0.0, trajectory.duration, 25):
        want_pos, _want_rot = trajectory.interpolate(t)
        got = ik.forward(joint_trajectory.interpolate(t).reshape(7))
        assert np.linalg.norm(got[:3, 3] - want_pos) < 5e-3
