"""Tests for `clear_franka.ik`.

The solve takes only tool poses, so every case here builds its waypoints with
forward kinematics from a joint trajectory and then checks what IK recovers from
the poses alone. The joint trajectory is never handed to the solver — it is only
the ground truth the reconstruction is compared against.

Two cases carry most of the weight: a demo that rotates the wrist, which breaks
any solver that resolves the redundancy by pinning q7, and one that swings the
arm around the base, which breaks one that pins the arm-plane angle instead.
"""

import numpy as np
import pytest
from franky import Affine
from franky.kinematics import forward_kinematics, swivel_angle

from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS
from clear_franka.geometry import pack_Rp
from clear_franka.ik import CartesianIK, ik_frame_from_episode

# Stand-in for the robot's flange-to-TCP offset; the URDF-derived one needs the
# robot description, and nothing here depends on the exact value.
F_T_EE = np.eye(4)
F_T_EE[2, 3] = 0.19

# The teleop reset configuration: a posture demos actually start from.
Q_HOME = np.array([-0.06217873, 0.11705305, -0.5995304, -2.72505689, 1.98229098, 1.78315043, 0.44959301])

LOWER = np.asarray(DEFAULT_LOWER_JOINT_LIMITS, dtype=float)
UPPER = np.asarray(DEFAULT_UPPER_JOINT_LIMITS, dtype=float)


def fk(joint_pos):
    poses = np.stack([forward_kinematics(q, f_t_ee=Affine(F_T_EE)).matrix for q in joint_pos])
    return poses[:, :3, 3], poses[:, :3, :3]


def wrist_spin(sweep_rad, n=300, base=Q_HOME):
    """A demo that only spins the tool about its own axis, by `sweep_rad`.

    j7 alone moves, so the wrist centre is stationary and the arm plane never
    turns: there is exactly one right answer here, and it is the demo itself.
    """
    s = np.linspace(0.0, 1.0, n)
    joint_pos = np.tile(np.asarray(base, dtype=float), (n, 1))
    joint_pos[:, 6] += sweep_rad * s
    return np.linspace(0.0, 3.0, n), joint_pos


def wrist_sweep(sweep_rad, n=300, base=Q_HOME):
    """A demo that spins the tool by `sweep_rad` while the arm also moves.

    The shoulder and elbow motion turns the arm plane too, so a solve that holds
    the plane steady cannot reproduce these joints exactly — only the tool path.
    """
    timestamps, joint_pos = wrist_spin(sweep_rad, n=n, base=base)
    s = np.linspace(0.0, 1.0, n)
    joint_pos[:, 1] += 0.25 * np.sin(np.pi * s)
    joint_pos[:, 3] += 0.20 * np.sin(np.pi * s)
    return timestamps, joint_pos


def solve(timestamps, joint_pos, **kwargs):
    """Convert the demo's tool path back into joints, seeded at its first sample."""
    ee_pos, ee_rot = fk(joint_pos)
    q_seed = kwargs.pop("q_seed", joint_pos[0])
    max_distance = kwargs.pop("max_distance", 0.2)
    ik = CartesianIK(f_t_ee=F_T_EE, **kwargs)
    return ik.solve_trajectory(
        ee_pos, ee_rot, q_seed=q_seed, timestamps=timestamps, max_distance=max_distance
    )


def cartesian_error(solved, joint_pos):
    """Max position (m) and orientation (rad) error of `solved` against the demo."""
    got_pos, got_rot = fk(solved)
    want_pos, want_rot = fk(joint_pos)
    pos_err = np.linalg.norm(got_pos - want_pos, axis=1).max()
    cos = [np.clip((np.trace(g.T @ w) - 1.0) / 2.0, -1.0, 1.0) for g, w in zip(got_rot, want_rot)]
    return pos_err, float(np.arccos(cos).max())


@pytest.mark.parametrize("sweep", [2.4, 2.55, -2.0])
def test_pure_wrist_rotation_is_taken_up_by_the_wrist(sweep):
    """Only j7 needs to move here, and only j7 should.

    A solver that pins q7 instead leaves the shoulder and elbow to produce the
    spin and runs them out of range partway through. Rotating the wrist is the
    smallest joint motion that turns the tool about its own axis, so a solver
    that simply takes the smallest motion gets this right by construction.
    """
    timestamps, joint_pos = wrist_spin(sweep)
    solved = solve(timestamps, joint_pos)

    assert np.ptp(solved[:, 6]) == pytest.approx(abs(sweep), abs=0.02)
    # The rest of the arm holds station, up to the posture drift a per-waypoint
    # solve accumulates over 300 steps.
    assert np.abs(solved[:, :6] - joint_pos[:, :6]).max() < 0.1


@pytest.mark.parametrize("sweep", [2.4, 2.55, -2.0])
def test_wrist_rotation_with_arm_motion_tracks_the_tool_path(sweep):
    timestamps, joint_pos = wrist_sweep(sweep)
    solved = solve(timestamps, joint_pos)

    pos_err, rot_err = cartesian_error(solved, joint_pos)
    assert pos_err < 1e-6
    assert rot_err < 1e-6
    assert np.ptp(solved[:, 6]) == pytest.approx(abs(sweep), abs=0.02)


def test_the_tool_path_is_reproduced_exactly():
    """Not merely within tolerance: the tolerances exist for waypoints the arm
    cannot quite reach, and an ordinary one should land on the pose outright."""
    timestamps, joint_pos = wrist_sweep(2.4)
    solved = solve(timestamps, joint_pos)
    pos_err, rot_err = cartesian_error(solved, joint_pos)
    assert pos_err < 1e-9
    # 1e-7, not 1e-9: `cartesian_error` recovers the angle through arccos, which
    # loses half its precision near zero. This is the metric's floor, not the solve's.
    assert rot_err < 1e-7


def test_solution_is_continuous():
    """No jumps: a 100 Hz demo must not step multiple radians in one sample."""
    timestamps, joint_pos = wrist_sweep(-2.0)
    solved = solve(timestamps, joint_pos)
    assert np.abs(np.diff(solved, axis=0)).max() < 0.05


def test_stays_inside_the_joint_limits_with_margin():
    timestamps, joint_pos = wrist_sweep(2.55)
    margin = 0.05
    solved = solve(timestamps, joint_pos, joint_limit_margin=margin)
    assert (solved >= LOWER + margin - 1e-9).all()
    assert (solved <= UPPER - margin + 1e-9).all()


def test_steers_away_from_a_joint_limit_it_starts_against():
    """A joint pinned on a limit is one the rest of the path cannot use, so the
    solve should walk it back off rather than converge onto the limit and stay."""
    timestamps, joint_pos = wrist_spin(1.0)
    seed = joint_pos[0].copy()
    seed[4] = UPPER[4] - 0.03  # j5 parked just inside its upper limit
    ee_pos, ee_rot = fk(joint_pos)
    solved = CartesianIK(f_t_ee=F_T_EE).solve_trajectory(
        ee_pos, ee_rot, q_seed=seed, timestamps=timestamps
    )
    assert UPPER[4] - solved[-1, 4] > UPPER[4] - solved[0, 4]


def test_starts_at_the_seed_configuration():
    """So the arm can be pre-positioned to solved[0] without a jump at t=0."""
    timestamps, joint_pos = wrist_sweep(2.4)
    solved = solve(timestamps, joint_pos)
    assert solved[0] == pytest.approx(joint_pos[0], abs=1e-3)


def test_without_a_seed_it_picks_a_configuration_off_the_first_waypoint():
    timestamps, joint_pos = wrist_sweep(2.4)
    solved = solve(timestamps, joint_pos, q_seed=None)

    pos_err, rot_err = cartesian_error(solved, joint_pos)
    assert pos_err < 1e-6
    assert rot_err < 1e-6
    # Nothing pins the posture, so the start need not match the demo's — but it
    # must be a legal configuration, and a roomy one.
    assert np.min(np.minimum(solved[0] - LOWER, UPPER - solved[0])) > 0.1


def test_max_distance_is_honoured():
    timestamps, joint_pos = wrist_sweep(2.4)
    solved = solve(timestamps, joint_pos, max_distance=0.05)
    assert np.abs(np.diff(solved, axis=0)).max() <= 0.05


def base_sweep(n=200):
    """A demo that swings the arm around the base, turning its own arm plane."""
    s = np.linspace(0.0, 1.0, n)
    joint_pos = np.tile(Q_HOME, (n, 1))
    joint_pos[:, 0] += 2.2 * s
    return np.linspace(0.0, 3.0, n), joint_pos


def test_a_wide_base_swing_is_followed():
    """The arm plane turns right through this one, which is what defeats a solver
    that resolves the redundancy by pinning that angle."""
    timestamps, joint_pos = base_sweep()
    solved = solve(timestamps, joint_pos)

    pos_err, rot_err = cartesian_error(solved, joint_pos)
    assert pos_err < 1e-6
    assert rot_err < 1e-6
    swivel = np.array([swivel_angle(q) for q in solved])
    assert np.ptp(swivel) > 1.0  # the plane travelled with the demo
    assert np.abs(np.diff(solved, axis=0)).max() < 0.05  # and stayed continuous doing it


def test_a_branch_jump_is_a_failure_not_a_silent_result():
    """Sampling a wide swing at 6 waypoints leaves gaps no arm can execute.

    The solver has to say so: answering with the nearest branch would hand the
    impedance controller a step of a radian or more to chase.
    """
    timestamps, joint_pos = base_sweep(n=6)
    with pytest.raises(RuntimeError, match=r"more than max_distance"):
        solve(timestamps, joint_pos)

    # The same path is fine once it is sampled finely enough to follow.
    timestamps, joint_pos = base_sweep(n=400)
    solved = solve(timestamps, joint_pos)
    assert np.abs(np.diff(solved, axis=0)).max() <= 0.2


@pytest.mark.parametrize("offset", [
    [1.5, 0, 0, 0, 0, 0, 0],   # a different base rotation
    [0, 0, -1.5, 0, 0, 0, 0],  # a different elbow roll
    [0, 0.8, 0, 0.5, 0, 0, 0],
])
def test_a_seed_from_an_unrelated_posture_still_solves(offset):
    """The seed picks a starting posture, it is not a constraint on the path.

    A seed whose arm plane suits nothing about the first waypoint — one read off
    a config file, say — must not fail the solve; the plane gets rechosen.
    """
    timestamps, joint_pos = wrist_spin(2.4)
    solved = solve(timestamps, joint_pos, q_seed=Q_HOME + np.asarray(offset, dtype=float))

    # Held to the solve's own acceptance tolerances rather than to exactness: the
    # point is that an unrelated seed still yields a usable trajectory.
    pos_err, rot_err = cartesian_error(solved, joint_pos)
    assert pos_err < 1e-3
    assert rot_err < 1e-2
    assert np.abs(np.diff(solved, axis=0)).max() < 0.05


def test_ik_frame_from_episode_reads_the_tool_offset_off_the_first_sample():
    _, joint_pos = wrist_spin(2.4)
    ee_pos, ee_rot = fk(joint_pos)
    episode = {"joint_pos": joint_pos, "ee_pos": ee_pos, "ee_rot": ee_rot}

    q_seed, f_t_ee = ik_frame_from_episode(episode)
    assert q_seed == pytest.approx(joint_pos[0])
    assert f_t_ee.matrix == pytest.approx(F_T_EE, abs=1e-9)


@pytest.mark.parametrize("missing", ["joint_pos", "ee_pos"])
def test_ik_frame_from_episode_declines_an_episode_without_usable_samples(missing):
    """A Cartesian-only episode is exactly the case that has to keep working."""
    _, joint_pos = wrist_spin(2.4)
    ee_pos, ee_rot = fk(joint_pos)
    episode = {"joint_pos": joint_pos, "ee_pos": ee_pos, "ee_rot": ee_rot}
    episode[missing] = np.full_like(episode[missing], np.nan)

    assert ik_frame_from_episode(episode) == (None, None)


def test_unreachable_waypoint_reports_where_it_failed():
    timestamps, joint_pos = wrist_sweep(0.0, n=10)
    ee_pos, ee_rot = fk(joint_pos)
    ee_pos[5] = [3.0, 0.0, 0.5]  # far outside the workspace

    with pytest.raises(RuntimeError, match=r"IK failed at step 5/10"):
        CartesianIK(f_t_ee=F_T_EE).solve_trajectory(
            ee_pos, ee_rot, q_seed=joint_pos[0], timestamps=timestamps
        )


def test_solve_reports_a_miss_instead_of_raising():
    """What a control loop needs: an unreachable pose is a result, not an exception,
    so the caller can hold or replan on its own terms."""
    _, joint_pos = wrist_spin(0.0, n=1)
    out_of_reach = np.eye(4)
    out_of_reach[:3, 3] = [3.0, 0.0, 0.5]

    solution = CartesianIK(f_t_ee=F_T_EE).solve(out_of_reach, seed=joint_pos[0])
    assert not solution.reached
    assert solution.position_error > 1.0
    assert (solution.joint_pos >= LOWER).all() and (solution.joint_pos <= UPPER).all()


def test_solve_reaches_a_pose_it_can_reach():
    _, joint_pos = wrist_spin(0.3, n=2)
    ee_pos, ee_rot = fk(joint_pos)
    ik = CartesianIK(f_t_ee=F_T_EE)

    solution = ik.solve(pack_Rp(ee_rot[1], ee_pos[1]), seed=joint_pos[0])
    assert solution.reached
    assert solution.position_error < 1e-9
    # It moved off the seed to get there, and says by how much — which is what a
    # caller rate-limits on.
    assert solution.joint_motion == pytest.approx(0.3, abs=1e-3)


def test_solve_accepts_an_affine_or_a_matrix():
    _, joint_pos = wrist_spin(0.3, n=2)
    ee_pos, ee_rot = fk(joint_pos)
    matrix = pack_Rp(ee_rot[1], ee_pos[1])
    ik = CartesianIK(f_t_ee=F_T_EE)

    assert ik.solve(matrix, seed=joint_pos[0]).joint_pos == pytest.approx(
        ik.solve(Affine(matrix), seed=joint_pos[0]).joint_pos
    )


def test_seed_configuration_picks_a_roomy_posture():
    _, joint_pos = wrist_spin(0.0, n=1)
    ee_pos, ee_rot = fk(joint_pos)
    ik = CartesianIK(f_t_ee=F_T_EE)

    seed = ik.seed_configuration(pack_Rp(ee_rot[0], ee_pos[0]))
    assert np.min(np.minimum(seed - LOWER, UPPER - seed)) > 0.1
    assert ik.solve(pack_Rp(ee_rot[0], ee_pos[0]), seed=seed).reached


def test_seed_configuration_is_none_when_nothing_reaches_the_pose():
    out_of_reach = np.eye(4)
    out_of_reach[:3, 3] = [3.0, 0.0, 0.5]
    assert CartesianIK(f_t_ee=F_T_EE).seed_configuration(out_of_reach) is None


# ── solve_trajectory_prefix ─────────────────────────────────────────────────────
# The control-loop counterpart: a live caller mid-rollout cannot act on an
# exception, so it gets the executable prefix plus a reason instead. These pin
# down that the prefix is exactly the part solve_trajectory would have produced
# before it gave up, and that the raising contract is unchanged.


def prefix(timestamps, joint_pos, **kwargs):
    ee_pos, ee_rot = fk(joint_pos)
    q_seed = kwargs.pop("q_seed", joint_pos[0])
    max_distance = kwargs.pop("max_distance", 0.2)
    ik = CartesianIK(f_t_ee=F_T_EE, **kwargs)
    return ik.solve_trajectory_prefix(
        ee_pos, ee_rot, q_seed=q_seed, timestamps=timestamps, max_distance=max_distance
    )


def test_prefix_returns_the_whole_path_and_no_reason_when_it_solves():
    timestamps, joint_pos = wrist_sweep(2.0, n=60)
    solved, reason = prefix(timestamps, joint_pos)

    assert reason is None
    assert solved.shape == (60, 7)
    # Identical to what the raising entry point produces.
    np.testing.assert_allclose(solved, solve(timestamps, joint_pos))


def test_prefix_stops_at_an_unreachable_waypoint_and_says_where():
    timestamps, joint_pos = wrist_sweep(0.0, n=10)
    ee_pos, ee_rot = fk(joint_pos)
    ee_pos[5] = [3.0, 0.0, 0.5]  # far outside the workspace

    solved, reason = CartesianIK(f_t_ee=F_T_EE).solve_trajectory_prefix(
        ee_pos, ee_rot, q_seed=joint_pos[0], timestamps=timestamps
    )

    # Waypoint 5 failed, so waypoints 0..4 are executable.
    assert solved.shape == (5, 7)
    assert reason is not None and "IK failed at step 5/10" in reason


def test_prefix_stops_before_a_branch_jump():
    timestamps, joint_pos = base_sweep(n=6)
    solved, reason = prefix(timestamps, joint_pos)

    assert reason is not None and "more than max_distance" in reason
    # Whatever it managed is short of the full path but still a valid trajectory.
    assert 0 <= len(solved) < 6


def test_prefix_rows_are_a_continuous_executable_trajectory():
    """The prefix must be usable as-is: continuous, and inside the limits."""
    timestamps, joint_pos = base_sweep(n=6)
    solved, reason = prefix(timestamps, joint_pos)

    assert reason is not None
    if len(solved) >= 2:
        step = np.abs(np.diff(solved, axis=0)).max()
        assert step <= 0.2, f"prefix contains a {step:.3f} rad jump"
    assert np.all(solved >= LOWER) and np.all(solved <= UPPER)


def test_prefix_is_empty_when_the_first_pose_is_unreachable_without_a_seed():
    ee_pos = np.array([[3.0, 0.0, 0.5], [3.1, 0.0, 0.5]])
    ee_rot = np.stack([np.eye(3)] * 2)

    solved, reason = CartesianIK(f_t_ee=F_T_EE).solve_trajectory_prefix(
        ee_pos, ee_rot, q_seed=None
    )

    assert solved.shape == (0, 7)
    assert reason is not None and "out of reach" in reason


def test_prefix_does_not_alias_uninitialized_buffer_rows():
    """The prefix is built from an np.empty buffer, so it must be copied out."""
    timestamps, joint_pos = wrist_sweep(0.0, n=10)
    ee_pos, ee_rot = fk(joint_pos)
    ee_pos[5] = [3.0, 0.0, 0.5]

    ik = CartesianIK(f_t_ee=F_T_EE)
    solved, _ = ik.solve_trajectory_prefix(ee_pos, ee_rot, q_seed=joint_pos[0])
    before = solved.copy()
    # A later solve reuses a fresh buffer; the handed-out prefix must not shift.
    ik.solve_trajectory_prefix(*fk(joint_pos), q_seed=joint_pos[0])
    np.testing.assert_array_equal(solved, before)
    assert np.all(np.isfinite(solved))


def test_solve_trajectory_still_raises_after_the_refactor():
    """solve_trajectory is now a wrapper; replay depends on it raising."""
    timestamps, joint_pos = base_sweep(n=6)
    with pytest.raises(RuntimeError, match=r"more than max_distance"):
        solve(timestamps, joint_pos)
