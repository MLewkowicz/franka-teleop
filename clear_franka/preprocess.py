"""Demonstration episode preprocessing — trim leading idle, smooth, retime.

Shared preprocessing is called from two places:
  * in-memory by `replay.py` before playback when `replay.preprocess=true`;
  * file-to-file by the standalone CLI `preprocess_demonstrations.py` for
    batch re-runs.

Pipeline (each step gated by explicit function arguments):
  1. trim   → drops leading/trailing stationary segments (`Trajectory.trim`)
  2. retime → TOPPRA: time-optimal traversal under (max_vel, max_accel).
              Runs on the SPARSE trimmed waypoints (~hundreds) — well-conditioned.
              Running it AFTER Ruckig (~17k dense waypoints) made TOPPRA's
              reachability solver fail with FailUncontrollable.
  3. smooth → Ruckig: dense, jerk-bounded trajectory at fixed dt. Operates on
              the (possibly retimed) waypoints from step 2 and absorbs any
              remaining jitter through its bounded-jerk profile.

Aligned non-joint fields (ee_pos, ee_rot, gripper_open, …) are first sliced to
the trim window, then re-sampled at the final post-smooth/retime timestamps:
linear for vectors, slerp for rotations, nearest-neighbour for binary signals.

`toppra` and `ruckig` are imported lazily here so missing dependencies fail the
steps that need them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from clear_franka.joint_trajectory import Trajectory

logger = logging.getLogger(__name__)


PREPROCESSING_VERSION = 1

# Datasets that must be re-sampled at the new timestamps. Each entry maps the
# h5 dataset name → interpolation kind ("linear" | "slerp" | "nearest").
# ee_pos / ee_rot are intentionally absent: they are recomputed via FK from
# the final smoothed joint_pos so they stay consistent with the joint data.
_RESAMPLE_KIND = {
    "cmd_linear_vel": "linear",
    "cmd_angular_vel": "linear",
    "robot_abs_time": "linear",
    "gripper_open": "nearest",
    "buttons": "nearest",
    "enabled": "nearest",
}


from clear_franka.franka import fk_ee_poses


def preprocess_episode(
    raw_h5_path: Path | str,
    out_h5_path: Path | str,
    *,
    trim_enabled: bool = True,
    trim_time_window: float = 0.3,
    trim_threshold: float = 0.01,
    retime_enabled: bool = False,
    retime_sample_uniform: bool = False,
    retime_path_tol: float | None = None,
    retime_max_joint_vel: np.ndarray | list[float] | None = None,
    retime_max_joint_accel: np.ndarray | list[float] | None = None,
    smooth_enabled: bool = True,
    smooth_max_joint_vel: np.ndarray | list[float] | None = None,
    smooth_max_joint_accel: np.ndarray | list[float] | None = None,
    smooth_max_joint_jerk: np.ndarray | list[float] | None = None,
    smooth_dt: float = 0.001,
    gripper_dwell_s: float = 0.0,
    params_metadata: dict[str, Any] | None = None,
) -> bool:
    """Read a raw demonstration episode, apply the preprocessing pipeline, write it.

    Returns True on success. Returns False only for explicit data-quality
    failures such as too-short episodes or velocity-bound violations. Dependency,
    load, processing, and write exceptions bubble up to the caller.
    """
    raw_h5_path = Path(raw_h5_path)
    out_h5_path = Path(out_h5_path)

    if not raw_h5_path.exists():
        logger.error("preprocess: raw episode does not exist: %s", raw_h5_path)
        return False

    raw = _load_episode(raw_h5_path)

    out_arrays = preprocess_episode_arrays(
        raw,
        trim_enabled=trim_enabled,
        trim_time_window=trim_time_window,
        trim_threshold=trim_threshold,
        retime_enabled=retime_enabled,
        retime_sample_uniform=retime_sample_uniform,
        retime_path_tol=retime_path_tol,
        retime_max_joint_vel=retime_max_joint_vel,
        retime_max_joint_accel=retime_max_joint_accel,
        smooth_enabled=smooth_enabled,
        smooth_max_joint_vel=smooth_max_joint_vel,
        smooth_max_joint_accel=smooth_max_joint_accel,
        smooth_max_joint_jerk=smooth_max_joint_jerk,
        smooth_dt=smooth_dt,
        gripper_dwell_s=gripper_dwell_s,
    )
    if out_arrays is None:
        return False

    # --- write out -------------------------------------------------------
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    _write_episode(
        out_h5_path,
        arrays=out_arrays,
        raw_attrs=raw["_attrs"],
        raw_camera_group=raw.get("_camera_timestamps"),
        preprocessing_params={} if params_metadata is None else dict(params_metadata),
        raw_basename=raw_h5_path.name,
    )

    logger.info(
        "preprocess: wrote %s (%d samples, %.2fs)",
        out_h5_path,
        out_arrays["joint_pos"].shape[0],
        float(out_arrays["timestamps"][-1]),
    )
    return True


def preprocess_episode_arrays(
    raw: dict[str, Any],
    *,
    trim_enabled: bool = True,
    trim_time_window: float = 0.3,
    trim_threshold: float = 0.01,
    retime_enabled: bool = False,
    retime_sample_uniform: bool = False,
    retime_path_tol: float | None = None,
    retime_max_joint_vel: np.ndarray | list[float] | None = None,
    retime_max_joint_accel: np.ndarray | list[float] | None = None,
    smooth_enabled: bool = True,
    smooth_max_joint_vel: np.ndarray | list[float] | None = None,
    smooth_max_joint_accel: np.ndarray | list[float] | None = None,
    smooth_max_joint_jerk: np.ndarray | list[float] | None = None,
    smooth_dt: float = 0.001,
    gripper_dwell_s: float = 0.0,
) -> dict[str, np.ndarray] | None:
    """Apply the preprocessing pipeline to an already-loaded episode.

    Returns processed arrays on success, or None for the same handled failures
    as preprocess_episode().
    """
    if raw["joint_pos"].shape[0] < 2:
        logger.error("preprocess: episode too short (%d samples)", raw["joint_pos"].shape[0])
        return None

    if np.isnan(raw["joint_pos"]).any():
        logger.error("preprocess: joint_pos contains NaN — refusing to process")
        return None

    # --- 1. trim ----------------------------------------------------------
    timestamps_in = raw["timestamps"].astype(np.float64)
    joint_pos_in = raw["joint_pos"].astype(np.float64)
    if trim_enabled:
        traj = Trajectory(joint_pos_in, timestamps_in)
        trimmed = traj.trim(
            time_window=float(trim_time_window),
            threshold=float(trim_threshold),
        )
        # Trajectory.trim slices the original arrays [lo:hi+1], so trimmed.waypts_time
        # values are exact members of timestamps_in. Recover the slice indices.
        lo = int(np.searchsorted(timestamps_in, trimmed.waypts_time[0], side="left"))
        hi_excl = int(np.searchsorted(timestamps_in, trimmed.waypts_time[-1], side="right"))
    else:
        lo = 0
        hi_excl = timestamps_in.shape[0]

    sliced = {k: v[lo:hi_excl] for k, v in raw.items() if isinstance(v, np.ndarray)}
    # Rebase timestamps so the trimmed segment starts at 0. Other aligned arrays
    # carry absolute robot time and shouldn't be rebased.
    times_trim = sliced["timestamps"] - sliced["timestamps"][0]
    joint_pos_trim = sliced["joint_pos"]
    trimmed_traj = Trajectory(joint_pos_trim, times_trim)

    logger.info(
        "preprocess: trim dropped %d leading + %d trailing samples (kept %d / %d, %.2fs)",
        lo,
        timestamps_in.shape[0] - hi_excl,
        hi_excl - lo,
        timestamps_in.shape[0],
        float(times_trim[-1]),
    )

    if trimmed_traj.num_waypts < 4:
        logger.error("preprocess: too few samples after trim (%d) — refusing to smooth", trimmed_traj.num_waypts)
        return None

    # --- 2. retime (TOPPRA) -----------------------------------------------
    # Run BEFORE smooth so TOPPRA operates on a sparse geometric path
    # (tens to hundreds of waypoints). Running it on Ruckig's dense 1 kHz
    # output (~17k samples) makes its reachability solver fail.
    current = trimmed_traj
    sparse_source_times: np.ndarray | None = None
    sparse_toppra_times: np.ndarray | None = None
    if retime_enabled:
        if retime_max_joint_vel is None and smooth_max_joint_vel is None:
            raise ValueError("retime_enabled=True requires retime_max_joint_vel or smooth_max_joint_vel")
        if retime_max_joint_accel is None and smooth_max_joint_accel is None:
            raise ValueError("retime_enabled=True requires retime_max_joint_accel or smooth_max_joint_accel")
        max_vel = np.asarray(
            smooth_max_joint_vel if retime_max_joint_vel is None else retime_max_joint_vel,
            dtype=np.float64,
        )
        max_accel = np.asarray(
            smooth_max_joint_accel if retime_max_joint_accel is None else retime_max_joint_accel,
            dtype=np.float64,
        )
        if retime_path_tol is not None:
            sparse = trimmed_traj.simplify(tol=float(retime_path_tol))
            logger.info(
                "preprocess: RDP simplified %d → %d waypoints (tol=%.4f rad)",
                trimmed_traj.num_waypts, sparse.num_waypts, retime_path_tol,
            )
        else:
            sparse = trimmed_traj

        # Capture source times (in the trimmed timeline) before TOPPRA retimes.
        # sparse_source_times[i] ↔ sparse_toppra_times[i]: same path index.
        sparse_source_times = np.asarray(sparse.waypts_time, dtype=np.float64)

        current = sparse.retime(
            max_vel=max_vel,
            max_accel=max_accel,
            sample_uniform=bool(retime_sample_uniform),
        )

        # Capture retimed timestamps before Ruckig overwrites current.waypts_time.
        sparse_toppra_times = np.asarray(current.waypts_time, dtype=np.float64)

    # --- 3. smooth (Ruckig) -----------------------------------------------
    # Operates on the (possibly retimed) waypoints. Ruckig's bounded-jerk
    # profile absorbs residual jitter and emits a dense control-rate stream.
    if smooth_enabled:
        if smooth_max_joint_vel is None:
            raise ValueError("smooth_enabled=True requires smooth_max_joint_vel")
        if smooth_max_joint_accel is None:
            raise ValueError("smooth_enabled=True requires smooth_max_joint_accel")
        if smooth_max_joint_jerk is None:
            raise ValueError("smooth_enabled=True requires smooth_max_joint_jerk")
        current = current.smooth(
            max_vel=np.asarray(smooth_max_joint_vel, dtype=np.float64),
            max_accel=np.asarray(smooth_max_joint_accel, dtype=np.float64),
            max_jerk=np.asarray(smooth_max_joint_jerk, dtype=np.float64),
            dt=float(smooth_dt),
        )

    # Final joint trajectory and timestamps.
    joint_pos_out = np.asarray(current.waypts, dtype=np.float64)
    times_out = np.asarray(current.waypts_time, dtype=np.float64)
    times_out = times_out - times_out[0]
    if times_out[-1] <= 0:
        logger.error("preprocess: degenerate output duration %.6f", float(times_out[-1]))
        return None

    target_times = times_out
    if sparse_source_times is not None and sparse_toppra_times is not None:
        # Map each dense output time to its source time in the trimmed trajectory.
        # The (sparse_toppra_times → sparse_source_times) mapping is monotone, so
        # np.interp gives an exact, principled alignment without nearest-neighbour
        # joint matching. For RDP: source times are exact members of times_trim.
        target_times = np.interp(times_out, sparse_toppra_times, sparse_source_times)
        target_times = np.clip(target_times, times_trim[0], times_trim[-1])

    # Velocity-bound assertion (post-smooth). Allow 5% slack for numerical diff.
    if smooth_enabled:
        max_vel_cfg = np.asarray(smooth_max_joint_vel, dtype=np.float64)
        joint_vel_check = np.diff(joint_pos_out, axis=0) / np.maximum(np.diff(times_out)[:, None], 1e-9)
        peak = np.abs(joint_vel_check).max(axis=0)
        if np.any(peak > max_vel_cfg * 1.05):
            logger.error(
                "preprocess: smoothed joint velocity exceeds 1.05× cap: peak=%s cap=%s",
                np.array2string(peak, precision=3),
                np.array2string(max_vel_cfg, precision=3),
            )
            return None

    # --- 4. resample aligned fields at the new timestamps -----------------
    resampled = _resample_aligned_fields(
        sliced,
        source_times=times_trim,
        target_times=target_times,
    )

    # --- 5. recompute ee_pos / ee_rot via FK on the smoothed joints ----------
    # The recorded Cartesian values would be inconsistent with the smoothed
    # joint positions; FK on joint_pos_out guarantees consistency.
    if "ee_pos" in sliced or "ee_rot" in sliced:
        fk_pos, fk_rot = fk_ee_poses(joint_pos_out)
        if "ee_pos" in sliced:
            resampled["ee_pos"] = fk_pos
        if "ee_rot" in sliced:
            resampled["ee_rot"] = fk_rot
        logger.info("preprocess: recomputed ee_pos/ee_rot via FK (%d frames)", joint_pos_out.shape[0])

    # --- 6. insert dwell at gripper state transitions ----------------------
    # TOPPRA allocates no time to stationary segments (zero geometric length),
    # so gripper open/close events get compressed to zero dwell. Re-insert
    # explicit hold frames so the physical gripper has time to complete its motion.
    if gripper_dwell_s > 0.0 and "gripper_open" in resampled:
        joint_pos_out, times_out, resampled = _insert_gripper_dwell(
            joint_pos_out, times_out, resampled,
            dwell_s=float(gripper_dwell_s), dt=float(smooth_dt),
        )

    # Recompute joint_vel from final joint_pos (after any dwell insertion).
    # np.gradient gives centred differences; covers the zero-velocity dwell
    # frames correctly since adjacent positions are identical there.
    if joint_pos_out.shape[0] >= 2:
        joint_vel_out = np.gradient(joint_pos_out, times_out, axis=0)
    else:
        joint_vel_out = np.zeros_like(joint_pos_out)

    out_arrays: dict[str, np.ndarray] = {
        "timestamps": times_out,
        "joint_pos": joint_pos_out,
        "joint_vel": joint_vel_out,
    }
    out_arrays.update(resampled)
    return out_arrays


def _insert_gripper_dwell(
    joint_pos: np.ndarray,
    times: np.ndarray,
    resampled: dict[str, np.ndarray],
    dwell_s: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Insert symmetric hold frames around every gripper open/close transition.

    For each index where gripper_open changes, inserts `half_n` frames holding
    the pre-change state immediately before the transition and `half_n` frames
    holding the post-change state immediately after. This makes the dwell work
    correctly in both forward and reverse playback.
    """
    gripper = resampled["gripper_open"]
    changes = np.where(gripper[1:] != gripper[:-1])[0] + 1
    if len(changes) == 0:
        return joint_pos, times, resampled

    half_n = max(1, round(dwell_s / dt / 2))
    logger.info(
        "preprocess: inserting %.2fs symmetric dwell (%d+%d frames) at %d gripper transition(s): indices %s",
        dwell_s, half_n, half_n, len(changes), changes.tolist(),
    )

    resampled = dict(resampled)

    def _repeat_after(arr: np.ndarray, idx: int, n: int) -> np.ndarray:
        tile = np.repeat(arr[idx : idx + 1], n, axis=0)
        return np.concatenate([arr[: idx + 1], tile, arr[idx + 1 :]])

    for idx in sorted(changes.tolist(), reverse=True):
        # Post-change: half_n frames holding the post-change state.
        t_post = times[idx] + np.arange(1, half_n + 1) * dt
        times = np.concatenate([times[: idx + 1], t_post, times[idx + 1 :] + half_n * dt])
        joint_pos = _repeat_after(joint_pos, idx, half_n)
        for key in list(resampled.keys()):
            resampled[key] = _repeat_after(resampled[key], idx, half_n)

        # Pre-change: half_n frames holding the pre-change state (at idx-1).
        if idx > 0:
            pre = idx - 1
            t_pre = times[pre] + np.arange(1, half_n + 1) * dt
            times = np.concatenate([times[: pre + 1], t_pre, times[pre + 1 :] + half_n * dt])
            joint_pos = _repeat_after(joint_pos, pre, half_n)
            for key in list(resampled.keys()):
                resampled[key] = _repeat_after(resampled[key], pre, half_n)

    return joint_pos, times, resampled


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_aligned_fields(
    sliced: dict[str, np.ndarray],
    *,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Re-sample non-joint aligned arrays from `source_times` onto `target_times`.

    `source_times` is the trimmed-and-rebased timeline of the original 1 kHz
    samples; `target_times` is the (possibly denser, possibly retimed) post-
    pipeline timeline that the joint trajectory now lives on.

    Kinds:
      * "linear" — np.interp per axis
      * "slerp"  — scipy Slerp on 3×3 rotation matrices
      * "nearest" — step interpolation, preserves binary semantics for
                    gripper_open / buttons / enabled
    """
    out: dict[str, np.ndarray] = {}
    # Clamp target times into the source range so interpolators don't extrapolate.
    # If Ruckig overshoots the input duration slightly, the tail samples just
    # repeat the final source value, which is the desired "hold-at-end" behaviour.
    target_clamped = np.clip(target_times, source_times[0], source_times[-1])

    for name, kind in _RESAMPLE_KIND.items():
        if name not in sliced:
            continue
        arr = sliced[name]
        if kind == "linear":
            out[name] = _interp_linear(arr, source_times, target_clamped)
        elif kind == "slerp":
            out[name] = _interp_slerp(arr, source_times, target_clamped)
        elif kind == "nearest":
            out[name] = _interp_nearest(arr, source_times, target_clamped)
    return out


def _interp_linear(arr: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.interp(dst_t, src_t, arr).astype(arr.dtype, copy=False)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((dst_t.shape[0], flat.shape[1]), dtype=arr.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, flat[:, j])
    return out.reshape((dst_t.shape[0],) + arr.shape[1:])


def _interp_slerp(arr: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """3×3 rotation matrices → Slerp → 3×3."""
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        # Fall back to linear if shape doesn't match expectations.
        return _interp_linear(arr, src_t, dst_t)
    rotations = Rotation.from_matrix(arr)
    slerp = Slerp(src_t, rotations)
    return slerp(dst_t).as_matrix().astype(arr.dtype, copy=False)


def _interp_nearest(arr: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Step-style nearest-neighbour lookup. Preserves binary signal integrity."""
    # `searchsorted` gives the insertion index; pick whichever neighbour is closer.
    idx = np.searchsorted(src_t, dst_t, side="left")
    idx = np.clip(idx, 0, src_t.shape[0] - 1)
    # For each target point, check whether idx-1 is closer than idx.
    left = np.clip(idx - 1, 0, src_t.shape[0] - 1)
    pick_left = (dst_t - src_t[left]) < (src_t[idx] - dst_t)
    chosen = np.where(pick_left, left, idx)
    return arr[chosen]
