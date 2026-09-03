"""Tests that preprocessing keeps gripper events at the pose they were demonstrated at.

The pipeline retimes the arm (TOPPRA on an RDP-simplified path, then Ruckig) but
carries `gripper_open` across by mapping output time back to demonstrated time.
If that map is wrong the arm still follows the right path — only the gripper
command slides along it, firing somewhere the operator never opened the gripper.

The episode built here is the shape that exposes it: fast straight transits, then
a slow, finely sampled descent to a "basin" pose where the gripper opens, then a
lift away. RDP keeps many waypoints in the slow detailed part and few in the long
transits, so a map that assumes the traversal covers one waypoint per equal slice
of time drifts badly — and drifts late, releasing after the lift has begun.
"""

import numpy as np
import pytest
import toppra as ta

from clear_franka.joint_trajectory import Trajectory
from clear_franka.preprocess import preprocess_episode_arrays

DOF = 7
DT = 0.01

# Limits loose enough that TOPPRA meaningfully compresses the demonstration.
MAX_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
MAX_ACCEL = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0])
MAX_JERK = np.array([50.0, 50.0, 50.0, 50.0, 60.0, 60.0, 60.0])


def _segment(q_from, q_to, duration, wiggle=None, lobes=1):
    """Waypoints along a joint-space segment, one per DT (excluding q_from).

    `wiggle` bows the segment off the straight line with `lobes` half-sines. RDP
    keeps waypoints where the path bends, so a wiggling segment survives
    simplification densely while a straight one collapses to its endpoints. That
    contrast is the point: it makes waypoint index advance very unevenly against
    traversal time, which is the condition the source-time map has to handle.
    """
    n = max(int(round(duration / DT)), 1)
    alphas = np.linspace(0.0, 1.0, n + 1)[1:]
    pts = q_from + alphas[:, None] * (q_to - q_from)
    if wiggle is not None:
        pts = pts + np.sin(lobes * np.pi * alphas)[:, None] * np.asarray(wiggle)
    return pts


def _basin_episode():
    """A reach-grasp-transit-release-retreat demo with the release at the basin.

    Long straight transits cover a lot of joint-space distance in two RDP
    waypoints; the short, wiggly descent into the basin and lift out of it cover
    very little distance in dozens. TOPPRA therefore burns through most of the
    waypoint indices in a small fraction of the traversal time — the same skew
    real demonstrations show, where the operator moves slowly and fussily around
    the grasp and the basin and quickly in between.

    Returns (episode dict, index of the gripper-open sample).
    """
    home = np.zeros(DOF)
    grasp = np.array([0.4, -0.3, 0.2, -1.9, 0.1, 1.5, 0.3])
    over_basin = np.array([-0.5, -0.2, 0.35, -1.7, -0.2, 1.4, 0.7])
    basin = over_basin + np.array([0.0, 0.09, 0.0, 0.10, 0.0, -0.04, 0.0])

    descend_wiggle = np.array([0.035, 0.0, -0.03, 0.025, 0.03, 0.0, -0.02])
    lift_wiggle = np.array([-0.03, 0.02, 0.028, -0.02, 0.0, 0.025, 0.015])

    parts = [
        home[None, :],
        _segment(home, grasp, 2.0),           # fast straight reach
        _segment(grasp, grasp, 0.5),          # settle, close gripper
        _segment(grasp, over_basin, 2.5),     # long straight transit
        _segment(over_basin, basin, 2.0, wiggle=descend_wiggle, lobes=7),
        _segment(basin, basin, 0.6),          # dwell — gripper opens here
    ]
    n_before_lift = sum(len(p) for p in parts)
    parts.append(_segment(basin, over_basin, 1.5, wiggle=lift_wiggle, lobes=5))
    parts.append(_segment(over_basin, home, 2.5))  # straight retreat

    joint_pos = np.concatenate(parts, axis=0)
    n = len(joint_pos)
    timestamps = np.arange(n) * DT

    # Open midway through the basin dwell, while the arm is still stationary.
    open_idx = n_before_lift - len(parts[5]) // 2
    close_idx = sum(len(p) for p in parts[:2]) + len(parts[2]) // 2

    gripper_open = np.ones(n)
    gripper_open[close_idx:open_idx] = 0.0

    return {
        "timestamps": timestamps,
        "joint_pos": joint_pos,
        "joint_vel": np.gradient(joint_pos, timestamps, axis=0),
        "gripper_open": gripper_open,
    }, open_idx


def _preprocess(raw, **overrides):
    kwargs = dict(
        trim_enabled=True,
        trim_time_window=0.3,
        trim_threshold=0.01,
        retime_enabled=True,
        retime_sample_uniform=True,
        retime_path_tol=0.001,
        retime_max_joint_vel=MAX_VEL,
        retime_max_joint_accel=MAX_ACCEL,
        smooth_enabled=True,
        smooth_max_joint_vel=MAX_VEL,
        smooth_max_joint_accel=MAX_ACCEL,
        smooth_max_joint_jerk=MAX_JERK,
        smooth_dt=0.001,
        gripper_dwell_s=0.0,
    )
    kwargs.update(overrides)
    return preprocess_episode_arrays(raw, **kwargs)


def _open_pose(out):
    """Joint configuration at the output frame where the gripper opens."""
    gripper = out["gripper_open"]
    opens = np.where((gripper[1:] == 1) & (gripper[:-1] == 0))[0] + 1
    assert len(opens) == 1, f"expected one open transition, got {len(opens)}"
    return out["joint_pos"][opens[0]]


@pytest.mark.parametrize("sample_uniform", [True, False])
def test_gripper_opens_at_the_demonstrated_pose(sample_uniform):
    raw, open_idx = _basin_episode()
    q_demo = raw["joint_pos"][open_idx]

    out = _preprocess(raw, retime_sample_uniform=sample_uniform)
    assert out is not None

    # 0.05 rad across all seven joints is far tighter than the ~0.5 rad the
    # basin-to-over-basin lift covers, so this fails if the event slides onto
    # the lift, and passes only if it stays at the bottom of the descent.
    assert np.linalg.norm(_open_pose(out) - q_demo) < 0.05


@pytest.mark.parametrize("sample_uniform", [True, False])
def test_gripper_event_is_not_biased_toward_the_lift(sample_uniform):
    """The failure mode was directional: late, i.e. displaced along the lift.

    Project the error onto the lift direction and require it to be small in
    absolute terms, so a regression that only ever adds delay is caught even if
    the unsigned distance were loosened.
    """
    raw, open_idx = _basin_episode()
    q_demo = raw["joint_pos"][open_idx]
    lift_dir = raw["joint_pos"][-1] - q_demo
    lift_dir = lift_dir / np.linalg.norm(lift_dir)

    out = _preprocess(raw, retime_sample_uniform=sample_uniform)
    along_lift = float((_open_pose(out) - q_demo) @ lift_dir)
    assert abs(along_lift) < 0.05


def test_retime_path_param_locates_each_output_sample_on_the_input_path():
    """`return_path_param` must say where on the input path each sample sits.

    This is the contract preprocess relies on to map retimed time back to
    demonstrated time; with sample_uniform=True, output index i is NOT input
    waypoint i, so the parameter has to be read from TOPPRA's parameterization.
    """
    raw, _ = _basin_episode()
    traj = Trajectory(raw["joint_pos"], raw["timestamps"]).simplify(tol=0.001)

    for sample_uniform in (True, False):
        retimed, s_out = traj.retime(
            max_vel=MAX_VEL,
            max_accel=MAX_ACCEL,
            sample_uniform=sample_uniform,
            return_path_param=True,
        )
        assert s_out.shape == (retimed.num_waypts,)
        assert np.all(np.diff(s_out) >= 0.0)
        assert s_out[0] == pytest.approx(0.0, abs=1e-6)
        assert s_out[-1] == pytest.approx(1.0, abs=1e-6)

        # Evaluating TOPPRA's geometric path at s_out[i] must reproduce output
        # sample i — that is precisely what "s_out[i] is where sample i sits on
        # the path" means, and it is false whenever s_out is faked from indices.
        path = ta.SplineInterpolator(
            np.linspace(0.0, 1.0, traj.num_waypts), traj.waypts, bc_type="clamped"
        )
        # 1e-3 rad leaves room for the linear interpolation of s(t) over the
        # solver grid against toppra's cubic parametrizer (~2e-4 here) while
        # still rejecting an index-derived s, which is off by ~1e-1 — larger
        # than the distance the arm covers between consecutive samples.
        assert np.abs(retimed.waypts - path(s_out)).max() < 1e-3

        # Retimed timestamps must stay strictly usable as an interpolation key.
        assert np.all(np.diff(retimed.waypts_time) > 0.0)
