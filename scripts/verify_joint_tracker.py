"""Offline verification for the joint-tracker IK (no robot required).

Run on a machine with the `ik` extra installed (jax + pyroki) and the Cortado
description available. CPU is fine and recommended:

    JAX_PLATFORMS=cpu uv run --extra ik python scripts/verify_joint_tracker.py
    JAX_PLATFORMS=cpu uv run --extra ik python scripts/verify_joint_tracker.py --episode data/episode_XXXX.h5

What it checks
--------------
1. FK consistency: CortadoIK.fk_ee (pyroki) vs clear_franka.franka.fk_ee_poses
   (yourdfpy). These must agree — both are the EE pose in the fr3_link0 frame.
   A constant offset here means the fr3_link0<->root frame conversion is wrong.
2. FK round-trip: random valid q -> fk_ee_poses -> CortadoIK.solve(seed=q) ->
   fk_ee_poses. Asserts position error < 1e-3 m and rotation geodesic < 1e-2 rad,
   and reports the IK cost distribution to calibrate deploy.ik.cost_threshold.
3. Continuity: solve_chained over a smooth Cartesian arc must not jump IK branch
   (max consecutive joint step below an elbow-flip threshold).
4. (optional) Real-episode dry-run: take a recorded ee_pos/ee_rot path (what the
   policy emits), solve_chained seeded from the recorded joint_pos[0], FK the
   result, and report pose error vs the recording + joint error vs recorded q.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    fk_ee_poses,
)
from clear_franka.cartesian_to_joint_ik import CortadoIK


def _rot_geodesic(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def _random_qs(n: int, rng: np.random.Generator, margin: float = 0.15) -> np.ndarray:
    lo = np.asarray(DEFAULT_LOWER_JOINT_LIMITS) + margin
    hi = np.asarray(DEFAULT_UPPER_JOINT_LIMITS) - margin
    return lo + rng.random((n, 7)) * (hi - lo)


def test_fk_consistency(ik: CortadoIK, rng) -> bool:
    qs = _random_qs(16, rng)
    pos_fk, rot_fk = fk_ee_poses(qs)
    pos_pk, rot_pk = ik.fk_ee(qs)
    dp = np.linalg.norm(pos_fk - pos_pk, axis=1)
    dr = np.array([_rot_geodesic(rot_fk[i], rot_pk[i]) for i in range(len(qs))])
    print(f"[1] FK consistency (pyroki vs yourdfpy): "
          f"max pos={dp.max()*1e3:.3f} mm, max rot={np.degrees(dr.max()):.3f} deg")
    ok = dp.max() < 1e-3 and dr.max() < 1e-2
    if not ok:
        print("    !! FK frames disagree — check T_root__fr3link0 conversion in CortadoIK.")
    return ok


def test_round_trip(ik: CortadoIK, rng) -> bool:
    qs = _random_qs(64, rng)
    pos, rot = fk_ee_poses(qs)
    pos_err, rot_err, costs = [], [], []
    for i in range(len(qs)):
        q_ik, cost = ik.solve(pos[i], rot[i], seed_q=qs[i])
        p2, r2 = fk_ee_poses(q_ik[None, :])
        pos_err.append(np.linalg.norm(p2[0] - pos[i]))
        rot_err.append(_rot_geodesic(r2[0], rot[i]))
        costs.append(float(cost))
    pos_err = np.array(pos_err); rot_err = np.array(rot_err); costs = np.array(costs)
    print(f"[2] FK round-trip over {len(qs)} poses: "
          f"pos err max={pos_err.max()*1e3:.3f} mm (p95={np.percentile(pos_err,95)*1e3:.3f}), "
          f"rot err max={np.degrees(rot_err.max()):.3f} deg")
    print(f"    IK cost: max={costs.max():.2e} p95={np.percentile(costs,95):.2e} "
          f"-> suggested deploy.ik.cost_threshold ~ {max(costs.max()*5, 1e-3):.1e}")
    ok = pos_err.max() < 1e-3 and rot_err.max() < 1e-2
    if not ok:
        print("    !! Round-trip out of tolerance.")
    return ok


def test_continuity(ik: CortadoIK, rng, step_thresh_rad: float = 0.5) -> bool:
    # Build a smooth Cartesian arc by interpolating two random reachable configs.
    qa, qb = _random_qs(2, rng)
    ts = np.linspace(0.0, 1.0, 25)
    q_path = (1 - ts)[:, None] * qa + ts[:, None] * qb  # smooth in joint space => smooth EE
    pos, rot = fk_ee_poses(q_path)
    q_ik, cost = ik.solve_chained(pos, rot, current_q=qa)
    steps = np.linalg.norm(np.diff(q_ik, axis=0), axis=1)
    print(f"[3] Continuity (chained over smooth arc): "
          f"max consecutive joint step={steps.max():.3f} rad, max cost={cost.max():.2e}")
    ok = steps.max() < step_thresh_rad
    if not ok:
        print(f"    !! Joint step exceeds {step_thresh_rad} rad — possible IK branch flip.")
    return ok


def test_episode_dryrun(ik: CortadoIK, path: str) -> bool:
    import h5py

    with h5py.File(path, "r") as f:
        ee_pos = np.asarray(f["ee_pos"])[:, :3]
        ee_rot = np.asarray(f["ee_rot"]).reshape(-1, 3, 3)
        joint_pos = np.asarray(f["joint_pos"])
    # Subsample to a policy-like horizon to keep it quick.
    n = len(ee_pos)
    idx = np.linspace(0, n - 1, min(n, 25)).astype(int)
    pos, rot, q_ref = ee_pos[idx], ee_rot[idx], joint_pos[idx]
    q_ik, cost = ik.solve_chained(pos, rot, current_q=joint_pos[0])
    p2, r2 = fk_ee_poses(q_ik)
    pos_err = np.linalg.norm(p2 - pos, axis=1)
    rot_err = np.array([_rot_geodesic(r2[i], rot[i]) for i in range(len(idx))])
    q_err = np.linalg.norm(q_ik - q_ref, axis=1)
    print(f"[4] Episode dry-run ({path}, {len(idx)} waypoints): "
          f"FK pos err max={pos_err.max()*1e3:.3f} mm, rot err max={np.degrees(rot_err.max()):.3f} deg")
    print(f"    joint error vs recorded q: max={q_err.max():.3f} rad (recorded q is one valid IK branch)")
    return pos_err.max() < 2e-3 and rot_err.max() < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None, help="optional episode_*.h5 for a real-data dry-run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("Building CortadoIK (loads Cortado URDF, builds pyroki robot, JIT warmup)...")
    ik = CortadoIK()
    ik.warmup()

    results = {
        "fk_consistency": test_fk_consistency(ik, rng),
        "round_trip": test_round_trip(ik, rng),
        "continuity": test_continuity(ik, rng),
    }
    if args.episode:
        results["episode_dryrun"] = test_episode_dryrun(ik, args.episode)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:16s}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
