"""Combined rotation + positional steering for the place stage.

Keeps the proven `TargetRotationSteering` (LangSteer) that tips the wrist into
the inverted-place basin (writes the rotation slice [3:9]) and adds a positional
branch driven by a *hardcoded* VoxPoser-style value map built from the workspace
bounding boxes (writes the position slice [0:3]). The position branch reuses
LangSteer's `PositionFieldGuidance` / `PositionTransform` math exactly — only the
value map is supplied locally (clear_franka.value_maps) instead of by the LLM
composer, so no `StageManager`/LLM is involved.

Both branches must share one `guidance_mode` because the policy routes a single
guidance fn (see policies/diffuser_actor_base._build_guidance_fns); we take it
from the rotation steering. `get_guidance` returns one tensor carrying the
rotation delta in [3:9] and the position delta in [0:3].

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
from steering.target_rotation import TargetRotationSteering

from clear_franka.value_maps import (
    RACK,
    build_place_value_map,
    front_face_center,
    gradient_field_tensor,
    load_boxes,
)

logger = logging.getLogger(__name__)


class CombinedBoxSteering(BaseSteering):
    """Rotation target + hardcoded value-map position steering in one module."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Rotation branch — unchanged proven steering (handles [3:9],
        # set_current_gripper_rotation, reset, set_rotation_scheduler).
        self._rot = TargetRotationSteering(cfg)
        # The policy routes by this single attribute; share it across branches.
        self.guidance_mode = self._rot.guidance_mode

        self.device = cfg.get("device", "cuda")
        self.horizon = int(cfg.get("horizon", 20))

        pcfg = dict(cfg.get("position", {}))
        ws_min = np.asarray(pcfg["workspace_bounds_min"], dtype=np.float32)
        ws_max = np.asarray(pcfg["workspace_bounds_max"], dtype=np.float32)
        map_size = int(pcfg.get("map_size", 100))

        boxes = load_boxes(pcfg["boxes_path"])
        face_thickness_m = float(pcfg.get("face_thickness_m", 0.04))
        forward_extend_m = float(pcfg.get("forward_extend_m", 0.06))
        basin_y_offset_m = float(pcfg.get("basin_y_offset_m", 0.0))
        basin_z_offset_m = float(pcfg.get("basin_z_offset_m", 0.0))
        rack_name = pcfg.get("rack", RACK)
        vm = build_place_value_map(
            boxes,
            ws_min=ws_min,
            ws_max=ws_max,
            map_size=map_size,
            avoidance_weight=float(pcfg.get("avoidance_weight", 2.0)),
            obstacle_sigma=float(pcfg.get("obstacle_sigma", 1.0)),
            suppress_affordance_in_obstacles=bool(
                pcfg.get("suppress_affordance_in_obstacles", False)),
            face_thickness_m=face_thickness_m,
            forward_extend_m=forward_extend_m,
            obstacle_back_extend_m=float(pcfg.get("obstacle_back_extend_m", 0.0)),
            wall_x_offset_m=float(pcfg.get("wall_x_offset_m", 0.0)),
            avoidance_carve_radius_m=float(pcfg.get("avoidance_carve_radius_m", 0.0)),
            affordance_y_extent_m=(
                float(pcfg["affordance_y_extent_m"])
                if pcfg.get("affordance_y_extent_m") is not None else None),
            affordance_z_extent_m=(
                float(pcfg["affordance_z_extent_m"])
                if pcfg.get("affordance_z_extent_m") is not None else None),
            include_underneath_wall=bool(
                pcfg.get("include_underneath_wall", True)),
            basin_y_offset_m=basin_y_offset_m,
            basin_z_offset_m=basin_z_offset_m,
            rack=rack_name,
        )
        self._value_map = vm
        self._grad_field = gradient_field_tensor(vm, self.device)

        # Basin target — center of the rack's front-face affordance slab. The
        # DistanceScaler ramps the position guidance DOWN as the EE approaches
        # this point, so the policy can settle into the basin without being
        # shoved past it.
        self._stage_target_world = front_face_center(
            boxes[rack_name]["center"], boxes[rack_name]["size"],
            face_thickness_m=face_thickness_m,
            forward_extend_m=forward_extend_m,
            y_offset_m=basin_y_offset_m,
            z_offset_m=basin_z_offset_m,
        )

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
        dist_cfg = dict(pcfg.get("distance", {}))
        self._pos = PositionFieldGuidance(
            horizon=self.horizon,
            guidance_strength=float(pcfg.get("guidance_strength", 0.05)),
            prediction_type="epsilon",
            guidance_mode=self.guidance_mode,
            start_guidance_timestep=int(pcfg.get("start_guidance_timestep", 10_000)),
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
                enabled=False, full_steps=0, decay_steps=80, floor=0.05),
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
        self._delta_norm_cap = float(pcfg.get("delta_norm_cap", 0.02))

        # Basin latch — once the *measured* EE first comes within this radius of
        # the basin, the position branch turns OFF for the rest of the stage and
        # stays off (rotation steering keeps running). The basin is an APPROACH
        # point, not the final rack-slot pose; the DistanceScaler alone ramps the
        # pull to its floor near the basin but is non-latching, so if the EE
        # drifts back out the pull re-engages and the EE loops in/out of the
        # basin while the policy tries to align with the rack. The latch hands
        # final alignment entirely to the policy once we've arrived. Set to 0 to
        # disable (fall back to pure DistanceScaler behavior). Cleared on reset().
        self._basin_latch_radius_m = float(pcfg.get("basin_latch_radius_m", 0.10))
        self._pos_latched = False

        # Log the first few pos_delta magnitudes so the user can see the field
        # actually nudging the trajectory (not destabilizing it). Tunable so we
        # can quiet it once the steering is tuned.
        self._diag_remaining = int(pcfg.get("diag_log_calls", 30))

        logger.info(
            "CombinedBoxSteering: mode=%s pos_strength=%s start_t=%s map_size=%d "
            "delta_norm_cap=%s ws=[%s, %s]",
            self.guidance_mode,
            pcfg.get("guidance_strength", 0.05),
            pcfg.get("start_guidance_timestep", 10_000),
            map_size,
            self._delta_norm_cap,
            np.array2string(ws_min, precision=2),
            np.array2string(ws_max, precision=2),
        )

    # ------------------------------------------------------------------
    # Wiring hooks (called by deploy._build_steering / the policy)
    # ------------------------------------------------------------------

    def set_rotation_scheduler(self, scheduler) -> None:
        self._rot.set_rotation_scheduler(scheduler)

    def set_position_scheduler(self, scheduler) -> None:
        self._pos_scheduler = scheduler

    def set_current_gripper_pos(self, gripper_pos: np.ndarray) -> None:
        self._coords.set_gripper_pos(np.asarray(gripper_pos, dtype=np.float32))

    def set_current_gripper_rotation(self, ee_euler_xyz: np.ndarray) -> None:
        self._rot.set_current_gripper_rotation(ee_euler_xyz)

    def reset(self) -> None:
        self._rot.reset()
        self._pos_latched = False

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
        """Rotation delta in [3:9] + value-map position delta in [0:3]."""
        # Rotation branch returns a full-size tensor (delta only in [3:9]).
        guidance = self._rot.get_guidance(
            current_sample, timestep, obs_embedding, model_output
        )

        # Basin latch — measured EE is fixed across this plan's denoising loop
        # (it's the observation pose), so this check is effectively per-plan.
        # Once we've arrived within the latch radius, drop the position branch
        # entirely (return rotation-only) and never re-engage it this stage.
        if self._basin_latch_radius_m > 0.0:
            ee = self._coords.current_gripper_pos
            if not self._pos_latched and ee is not None:
                basin = torch.as_tensor(
                    self._stage_target_world, dtype=ee.dtype, device=ee.device)
                d = float(torch.norm(ee - basin).item())
                if d <= self._basin_latch_radius_m:
                    self._pos_latched = True
                    logger.info(
                        "CombinedBoxSteering: basin latch ENGAGED (d=%.3fm <= "
                        "%.3fm) — position steering OFF for the rest of the stage; "
                        "rotation steering continues, policy handles rack alignment.",
                        d, self._basin_latch_radius_m,
                    )
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
            steps_in_stage=0,
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
            raw_max = float(pos_delta.abs().max().item())
            if self._delta_norm_cap > 0.0:
                pos_delta = self._delta_norm_cap * torch.tanh(
                    pos_delta / self._delta_norm_cap
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
