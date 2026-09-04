"""Positional steering for the place stage.

A VoxPoser-style value-map position branch (writes the position slice [0:3]
of the guidance tensor) driven by LangSteer's `PositionFieldGuidance` /
`PositionTransform` math — only the value map is supplied locally, so no
`StageManager` is involved. There is no rotation branch: the task's LLM
planner only ever synthesizes a *position* target (see
`clear_franka/value_map_llm/prompts/franka/planner_prompt.txt` — grasp and
final orientation are explicitly the base policy's job), so there is no
per-task orientation target to steer toward, and the policy is left fully in
control of rotation.

The position value map is always the LLM-synthesized one: a task string +
SAM 3 scene boxes -> a .npz (clear_franka.value_map_llm), loaded from
`cfg["llm"]["artifact_path"]`. No model loading or API call happens in the
control path here — that already happened when the artifact was written
(`deploy_diffuser_actor._synthesize_llm_value_map`, or offline via
`synthesize_value_map.py`).

This module is attached for every deploy stage (grasp AND place), never
swapped for None at a stage boundary — `set_deploy_stage(stage_idx)` (called
before every policy.forward()) just flips whether `get_guidance` returns real
guidance or constant zero for the current stage (`stage_indices` in config,
place-only by default). Staying attached across the grasp->place transition
keeps its internal state (position-stage progression) warm instead of
cold-starting exactly when place begins, and needs no timestep-threshold
delay to suppress guidance during grasp.

LangSteer must be on sys.path before this module is imported (the deploy script
calls _wire_langsteer first).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from core.steering import BaseSteering
from steering.coordinates import PositionTransform
from steering.diffusion_utils import get_alpha_bar
from steering.position_field import PositionFieldGuidance
from steering.scalers import (
    DistanceScaler,
    ScalerContext,
    StepScaler,
    TimestepScaler,
)

from clear_franka.value_maps import gradient_field_tensor

logger = logging.getLogger(__name__)


class PositionFieldSteering(BaseSteering):
    """Value-map position steering for the place stage (no rotation branch)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self._llm_task = ""  # set below from the artifact's metadata

        # The policy routes by this single attribute.
        self.guidance_mode = str(cfg.get("guidance_mode", "epsilon"))

        self.device = cfg.get("device", "cuda")
        self.horizon = int(cfg.get("horizon", 20))

        # Which deploy stage(s) (grasp=0 / place=1, ...) this module actually
        # steers; set_deploy_stage() flips `_deploy_stage_active` on every
        # policy.forward() call (see deploy_diffuser_actor._start_inference_
        # worker). Outside those stages get_guidance returns zero guidance —
        # this module is attached and called every step regardless of stage
        # (never swapped for None at the call site), so its internal state
        # (position-stage progression) stays warm across the grasp->place
        # transition instead of being rebuilt cold, and there is no
        # timestep-threshold delay to wait out — grasp is just a stage where
        # the guidance is constant zero everywhere.
        self._active_stage_indices = {int(s) for s in cfg.get("stage_indices", [1])}
        self._deploy_stage_active = False

        # Counts replans (policy.forward() calls, one per set_deploy_stage())
        # since the steered stage was last (re)entered -- feeds StepScaler so
        # guidance_strength can start high and ramp off quickly, instead of
        # pulling at full strength for the whole stage. Reset to 0 on entry
        # (see set_deploy_stage) and on episode reset.
        self._stage_steps = 0

        # `position` carries shared PositionFieldGuidance tuning (guidance
        # strength, delta cap, distance ramp, ...); `llm` (below) can override
        # any of them per-task, falling back to `position` for the rest (_pp).
        pcfg = dict(cfg.get("position", {}))

        # Value maps synthesized from the task + SAM 3 scene boxes (either
        # offline via synthesize_value_map.py, or live at deploy launch via
        # deploy_diffuser_actor._synthesize_llm_value_map) -> .npz. The
        # artifact carries its own grid — bounds derived from the detected
        # scene — so adopt them here: PositionTransform and
        # PositionFieldGuidance below must index the SAME grid the maps were
        # built on, or every gradient lookup is silently offset.
        from clear_franka.value_map_llm.artifact import load_stages

        lcfg = dict(cfg.get("llm", {}))
        pos_params = lcfg
        synth_stages, meta = load_stages(lcfg["artifact_path"])
        ws_min = np.asarray(meta["workspace_bounds_min"], dtype=np.float32)
        ws_max = np.asarray(meta["workspace_bounds_max"], dtype=np.float32)
        map_size = int(meta["map_size"])
        self._llm_task = str(meta.get("task", ""))

        self._pos_stages = [
            {
                "vm": st.value_map,
                "grad": gradient_field_tensor(st.value_map, self.device),
                "target": st.target_world,
                "radius": st.arrival_radius_m,
                "label": st.label,
            }
            for st in synth_stages
        ]
        self._pos_stage_idx = 0
        vm = self._pos_stages[0]["vm"]
        self._stage_target_world = self._pos_stages[0]["target"]
        logger.info(
            "PositionFieldSteering: LLM value maps from %s — task=%r, "
            "%d stage(s): %s",
            lcfg["artifact_path"], self._llm_task, len(self._pos_stages),
            [(st["label"], np.round(st["target"], 3).tolist())
             for st in self._pos_stages],
        )

        def _pp(key, default):
            """LLM-block position param with fallback to the shared block."""
            return pos_params.get(key, pcfg.get(key, default))

        self._value_map = vm
        self._grad_field = gradient_field_tensor(vm, self.device)

        # gripper_loc_bounds + relative come from the policy yaml (merged in by
        # deploy._build_steering). They drive the model<->world transform that
        # maps the predicted (relative, normalized) positions into the world
        # frame the value map lives in.
        self._coords = PositionTransform(
            gripper_loc_bounds=cfg.get("gripper_loc_bounds", None),
            workspace_min=ws_min,
            workspace_max=ws_max,
            is_relative=bool(cfg.get("relative", True)),
            device=self.device,
        )

        # DistanceScaler ramps the guidance DOWN as the EE approaches the
        # basin: full strength when d ≥ `full`, linear ramp to `floor` at d ≤
        # `near`. Keeps the pull strong while the policy is still far away and
        # lets the model settle once it's in the basin (cribbed from
        # VoxPoserSteering's own ctx wiring).
        dist_cfg = dict(_pp("distance", {}))
        # StepScaler ramps guidance_strength DOWN over replans spent in the
        # steered stage (self._stage_steps) -- lets the pull start strong at
        # stage entry (steps_in_stage=0) and quickly hand control back to the
        # policy instead of pulling at full strength the whole stage.
        step_cfg = dict(_pp("step", {}))
        self._pos = PositionFieldGuidance(
            horizon=self.horizon,
            guidance_strength=float(_pp("guidance_strength", 0.05)),
            prediction_type="epsilon",
            guidance_mode=self.guidance_mode,
            start_guidance_timestep=int(_pp("start_guidance_timestep", 10_000)),
            coordinates=self._coords,
            map_size=map_size,
            # TimestepScaler ramps from min_scale at low t up to 1.0 at high t —
            # the opposite of what we want here, since dps_coeff = 1/√ᾱ already
            # blows up at high t. Leave it off and rely on `delta_norm_cap` to
            # bound per-step magnitude.
            timestep_scaler=TimestepScaler(enabled=False, min_scale=0.1),
            distance_scaler=DistanceScaler(
                enabled=bool(dist_cfg.get("enabled", True)),
                full=float(dist_cfg.get("full", 0.30)),
                near=float(dist_cfg.get("near", 0.05)),
                floor=float(dist_cfg.get("floor", 0.10)),
            ),
            step_scaler=StepScaler(
                enabled=bool(step_cfg.get("enabled", False)),
                full_steps=int(step_cfg.get("full_steps", 0)),
                decay_steps=int(step_cfg.get("decay_steps", 80)),
                floor=float(step_cfg.get("floor", 0.05)),
            ),
        )

        # PositionFieldGuidance.compute() reads only .value_map + .gradient_field.
        self._stage = SimpleNamespace(value_map=vm, gradient_field=self._grad_field)
        self._pos_scheduler: Any = None

        # Per-step L∞-norm cap on the position delta (in model space). The raw
        # `coeff · grad_model` can be huge (dps_coeff = 1/√ᾱ at high t, cost
        # gradients ~5–10 in model space, accumulated over 25 denoising steps);
        # without a cap, the trajectory pegs against the [-1, 1] normalization
        # bounds and lands miles outside the workspace. Cap preserves the
        # descent direction but limits magnitude; tune via cfg.position.
        self._delta_norm_cap = float(_pp("delta_norm_cap", 0.02))

        # The EE advances through `self._pos_stages` as it arrives at each
        # stage's target (within its own `arrival_radius_m`, set per task by
        # the planner LLM); after the last stage the position branch latches
        # off (there's no rotation branch left to keep running). See the
        # multi-stage advance/latch logic in get_guidance.
        self._pos_latched = False

        # Log the first few pos_delta magnitudes so the user can see the field
        # actually nudging the trajectory (not destabilizing it). Tunable so we
        # can quiet it once the steering is tuned.
        self._diag_remaining = int(_pp("diag_log_calls", 30))

        logger.info(
            "PositionFieldSteering: mode=%s target=%s pos_strength=%s "
            "start_t=%s map_size=%d delta_norm_cap=%s ws=[%s, %s]",
            self.guidance_mode,
            np.array2string(np.asarray(self._stage_target_world), precision=3),
            _pp("guidance_strength", 0.05),
            _pp("start_guidance_timestep", 10_000),
            map_size,
            self._delta_norm_cap,
            np.array2string(ws_min, precision=2),
            np.array2string(ws_max, precision=2),
        )

    # ------------------------------------------------------------------
    # Wiring hooks (called by deploy._build_steering / the policy)
    # ------------------------------------------------------------------

    def set_position_scheduler(self, scheduler) -> None:
        self._pos_scheduler = scheduler

    def set_deploy_stage(self, stage_idx: int) -> None:
        """Tell this (always-attached) module which deploy stage is active.

        Call before every policy.forward(), regardless of stage — get_guidance
        returns constant zero guidance for stages outside
        `stage_indices` (grasp by default) instead of the module being
        swapped for None at the call site.
        """
        active = int(stage_idx) in self._active_stage_indices
        if active:
            # Reset on (re)entry so the strength ramp restarts from full
            # strength each time the steered stage begins; otherwise count
            # one more replan spent in it.
            self._stage_steps = 0 if not self._deploy_stage_active else self._stage_steps + 1
        self._deploy_stage_active = active

    def set_current_gripper_pos(self, gripper_pos: np.ndarray) -> None:
        self._coords.set_gripper_pos(np.asarray(gripper_pos, dtype=np.float32))

    def reset(self) -> None:
        self._pos_latched = False
        self._stage_steps = 0
        # Rewind the position-stage sequence to the first target.
        self._pos_stage_idx = 0
        self._stage.value_map = self._pos_stages[0]["vm"]
        self._stage.gradient_field = self._pos_stages[0]["grad"]
        self._stage_target_world = self._pos_stages[0]["target"]

    # Lifecycle no-ops for run_experiment / policy compatibility.
    def setup_episode(self, task_name: str):
        return None, None

    def increment_step(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Guidance
    # ------------------------------------------------------------------

    def get_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        obs_embedding: Any,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Value-map position delta in [0:3]; rotation is left to the policy."""
        container = model_output if self.guidance_mode != "dps" else current_sample
        guidance = torch.zeros_like(container)
        # Outside the steered deploy stage(s) (grasp by default) — constant
        # zero guidance. Set by set_deploy_stage() before every
        # policy.forward() call.
        if not self._deploy_stage_active:
            return guidance

        # Position-stage progression / latch. The measured EE is fixed across
        # this plan's denoising loop (it's the observation pose), so these
        # checks are effectively per-plan. Advance through the position-stage
        # sequence as the EE arrives at each target (within its own
        # arrival_radius_m); latch off after the last. One-way (monotonic).
        ee = self._coords.current_gripper_pos
        if not self._pos_latched and ee is not None:
            cur = self._pos_stages[self._pos_stage_idx]
            d = float(torch.norm(
                ee - torch.as_tensor(cur["target"], dtype=ee.dtype, device=ee.device)
            ).item())
            if cur["radius"] > 0.0 and d <= cur["radius"]:
                if self._pos_stage_idx + 1 < len(self._pos_stages):
                    self._pos_stage_idx += 1
                    nxt = self._pos_stages[self._pos_stage_idx]
                    self._stage.value_map = nxt["vm"]
                    self._stage.gradient_field = nxt["grad"]
                    self._stage_target_world = nxt["target"]
                    logger.info(
                        "PositionFieldSteering: position stage %d reached "
                        "(d=%.3fm) → advancing to stage %d target=%s",
                        self._pos_stage_idx, d, self._pos_stage_idx + 1,
                        np.array2string(nxt["target"], precision=3))
                else:
                    self._pos_latched = True
                    logger.info(
                        "PositionFieldSteering: final position stage reached "
                        "(d=%.3fm) — steering OFF for the rest of the stage.", d)
        if self._pos_latched:
            return guidance

        alpha_bar = get_alpha_bar(self._pos_scheduler, timestep, device=self.device)
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        num_train_t = (
            self._pos_scheduler.config.num_train_timesteps
            if self._pos_scheduler is not None
               and hasattr(self._pos_scheduler, "config")
            else None
        )
        ctx = ScalerContext(
            timestep=t,
            num_train_timesteps=num_train_t,
            ee_pos=self._coords.current_gripper_pos,
            stage_target=self._stage_target_world,  # basin -> DistanceScaler ramps off
            steps_in_stage=self._stage_steps,
        )
        pos_delta = self._pos.compute(
            x_t=current_sample,
            eps=model_output,
            timestep=timestep,
            alpha_bar=alpha_bar,
            stage=self._stage,
            ctx=ctx,
            episode_step=0,
        )
        if pos_delta is not None:
            # LangSteer's PositionFieldGuidance computes `delta = +coeff *
            # ∇cost`. In epsilon mode, adding `+delta` to ε shifts x₀ by
            # `−(√(1−ᾱ)/√ᾱ)·delta` via Tweedie's chain rule — i.e. in the
            # `−∇cost` direction (descent), which is correct. In DPS mode the
            # delta is added DIRECTLY to x_{t-1} after the scheduler step, so
            # `+∇cost` ascends the cost map (wrong sign). LangSteer's own
            # configs only use epsilon for VoxPoser, so this dps-mode sign
            # error is dormant upstream — flip it here for the dps route.
            if self.guidance_mode == "dps":
                pos_delta = -pos_delta

            # Soft (tanh) per-element saturator. Smoothly bounds large deltas
            # at ±cap without globally rescaling: small deltas pass through
            # essentially unchanged, so the DistanceScaler's ramp toward the
            # basin remains visible in the executed magnitude. (A hard global
            # rescale would re-saturate everything at the cap and mask the
            # ramp.) Tanh preserves the per-element sign and direction.
            raw_pos_delta = pos_delta
            raw_max = float(raw_pos_delta.abs().max().item())
            if self._delta_norm_cap > 0.0:
                pos_delta = self._delta_norm_cap * torch.tanh(
                    raw_pos_delta / self._delta_norm_cap
                )

            if self._diag_remaining > 0:
                self._diag_remaining -= 1
                t_int = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
                ee = self._coords.current_gripper_pos
                d = (float(torch.norm(ee - torch.tensor(self._stage_target_world,
                                                       dtype=ee.dtype, device=ee.device)).item())
                     if ee is not None else float("nan"))
                dist_scale = self._pos._distance_scaler.compute(ctx)
                # Frame-direction sanity check. The applied pos_delta is in
                # MODEL space; world→model is a per-axis POSITIVE scaling, so
                # sign(model_delta) == sign(world_delta) per axis. Aggregating
                # the delta across the 20 waypoints averages out the
                # per-waypoint Tweedie scatter and gives a stable "where is the
                # steering pushing the trajectory on net" direction in world
                # frame. Compare to the direction toward the basin: when the
                # EE is well outside the basin (dist_scale ≈ 1), the average
                # net push should at least roughly point toward the rack
                # (signs match per axis). Persistent sign mismatch on the same
                # axis over many calls = real frame/sign bug.
                ee_world = (ee.cpu().numpy() if ee is not None
                            else np.array([np.nan]*3, dtype=np.float32))
                mean_model_delta = pos_delta[0].mean(dim=0).detach().cpu().numpy()
                to_basin = self._stage_target_world - ee_world
                tn = np.linalg.norm(to_basin)
                to_basin_unit = to_basin / tn if tn > 1e-9 else to_basin
                mn = np.linalg.norm(mean_model_delta)
                mean_delta_unit = mean_model_delta / mn if mn > 1e-9 else mean_model_delta
                sign_match = [
                    bool(np.sign(to_basin[i]) == np.sign(mean_model_delta[i])
                         or abs(mean_model_delta[i]) < 1e-6)
                    for i in range(3)
                ]
                logger.info(
                    "  pos_delta t=%d  d_to_basin=%.3fm  dist_scale=%.2f  "
                    "raw_max=%.4f  cap=%.4f  applied_max=%.4f  applied_mean=%.4f",
                    t_int, d, dist_scale,
                    raw_max, self._delta_norm_cap,
                    float(pos_delta.abs().max()),
                    float(pos_delta.abs().mean()),
                )
                # Saturation diagnosis: raw_max alone can't distinguish "one
                # outlier waypoint" from "the whole horizon is pegged against
                # the cap" (the latter reads as noisy/jerky since a small
                # shift in the raw gradient's magnitude near the cap boundary
                # flips a large chunk of the applied delta). sat_frac is the
                # share of (batch*horizon*axis) elements whose raw magnitude
                # exceeds the cap (guaranteed tanh-saturated); per-axis raw
                # max shows whether one axis (usually the one aligned with
                # the dominant obstacle wall / affordance slab normal)
                # dominates the saturation.
                raw_abs = raw_pos_delta.abs()
                sat_frac = float((raw_abs > self._delta_norm_cap).float().mean().item())
                raw_flat = raw_abs.flatten()
                p50, p90, p99 = (
                    float(torch.quantile(raw_flat, q).item())
                    for q in (0.50, 0.90, 0.99)
                )
                raw_axis_max = raw_abs.amax(dim=(0, 1)).detach().cpu().numpy()
                applied_axis_max = pos_delta.abs().amax(dim=(0, 1)).detach().cpu().numpy()
                logger.info(
                    "  pos_delta saturation  sat_frac=%.2f  raw_p50=%.4f  "
                    "raw_p90=%.4f  raw_p99=%.4f  raw_axis_max(xyz)=%s  "
                    "applied_axis_max(xyz)=%s",
                    sat_frac, p50, p90, p99,
                    np.array2string(raw_axis_max, precision=4),
                    np.array2string(applied_axis_max, precision=4),
                )
                logger.info(
                    "  pos_delta frame-check  ee=%s  basin=%s  to_basin_unit=%s  "
                    "mean_delta_unit=%s  mean_delta=%s  sign_match(xyz)=%s",
                    np.array2string(ee_world, precision=3),
                    np.array2string(self._stage_target_world, precision=3),
                    np.array2string(to_basin_unit, precision=2),
                    np.array2string(mean_delta_unit, precision=2),
                    np.array2string(mean_model_delta, precision=5),
                    sign_match,
                )

            h = pos_delta.shape[1]
            guidance[:, :h, :3] = guidance[:, :h, :3] + pos_delta
        return guidance
