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
from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from steering.diffusion_utils import get_alpha_bar
from steering.position_field import PositionFieldGuidance
from steering.scalers import (
    DistanceScaler,
    ScalerContext,
    StepScaler,
    TimestepScaler,
)
from steering.target_rotation import TargetRotationSteering, _euler_to_matrix

from clear_franka.value_maps import (
    RACK,
    build_center_attractor_value_map,
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

        # Place mode selects the whole steering profile:
        #   rack    (mode B) — inverted glass onto the wine rack: near-180° flip
        #     rotation target + front-face affordance + obstacle walls.
        #   cabinet (mode A) — upright glass into the cabinet: upright rotation
        #     target + a gentle point-attractor at the cabinet placement point
        #     (no walls). The `cabinet` block overrides target_euler /
        #     rot_reverse_direction and supplies `cabinet.position`; everything
        #     else (scene bounds, gripper bounds, scalers, SLERP machinery) is
        #     shared. Switch modes with one line: `place_mode: rack|cabinet`.
        self._place_mode = str(cfg.get("place_mode", "rack")).lower()
        ccfg = dict(cfg.get("cabinet", {}))
        if self._place_mode == "cabinet":
            cfg = dict(cfg)  # shallow copy; override the rotation knobs the branches read
            if ccfg.get("target_euler") is not None:
                cfg["target_euler"] = ccfg["target_euler"]
            cfg["rot_reverse_direction"] = bool(ccfg.get("rot_reverse_direction", False))
            if ccfg.get("guidance_strength") is not None:
                cfg["guidance_strength"] = ccfg["guidance_strength"]

        # Rotation branch — TargetRotationSteering is kept for its bookkeeping
        # (target_euler → R_target_world, set_current_gripper_rotation → the live
        # relative target_6d, set_rotation_scheduler, guidance_mode). When
        # `rot_use_slerp` is on we DON'T call its linear get_guidance; instead we
        # reuse its target_6d + scheduler and steer along the SO(3) GEODESIC.
        self._rot = TargetRotationSteering(cfg)
        # The policy routes by this single attribute; share it across branches.
        self.guidance_mode = self._rot.guidance_mode

        self.device = cfg.get("device", "cuda")
        self.horizon = int(cfg.get("horizon", 20))

        # --- Geodesic (SLERP) rotation steering -----------------------------
        # The inverted-place target is a near-180° flip from the grasp pose. The
        # linear-6D pull in TargetRotationSteering can't choose which way the
        # wrist goes around it (and is ill-conditioned near the antipode). SLERP
        # per-horizon targets travel the great-circle geodesic; `rot_reverse_
        # direction` selects which way around: False = short way (clean geodesic),
        # True = forced LONG way (opposite wrist direction, same final pose).
        # NOTE: we can't use RotationFieldGuidance's `hemisphere_fix` for this —
        # it only avoids paths >180°, it can't FORCE the long way when the short
        # path is <180° (our 176° flip), so reversing requires negating the
        # target quaternion ourselves (see _slerp_targets). The per-horizon
        # geodesic targets feed the SAME proven dps/epsilon delta formula
        # TargetRotationSteering uses (no new sign/scaling surprises).
        self._rot_use_slerp = bool(cfg.get("rot_use_slerp", False))
        self._rot_reverse_direction = bool(cfg.get("rot_reverse_direction", False))
        self._rot_alpha_floor = float(cfg.get("ramp_floor", 0.0))
        self._rot_alpha_max = float(cfg.get("rot_horizon_alpha_max", 0.5))
        # Rotation pull strength (top-level `guidance_strength`, same value the
        # linear TargetRotationSteering branch uses).
        self.guidance_strength_rot = float(cfg.get("guidance_strength", 0.3))

        # Committed world-frame rotation axis (set on the first
        # set_current_gripper_rotation after reset). The long/short choice each
        # plan is then made to KEEP rotating about this fixed axis, so replanning
        # (and the shrinking remaining-angle) can't flip the direction. `_rot_negate`
        # is the per-plan decision (recomputed in set_current_gripper_rotation).
        self._rot_commit_axis: np.ndarray | None = None
        self._rot_negate = False
        # Only force the long way while the remaining rotation exceeds this (rad).
        # Below it, always short way — stops a tiny residual (whose axis can flip
        # vs the committed axis) from triggering a ~360° spin at the basin.
        self._rot_reverse_min_angle = float(cfg.get("rot_reverse_min_angle_rad", 1.2))

        # When True, reaching the basin / final position stage turns off ALL
        # steering (rotation too), not just the position branch. `_all_off` is the
        # latched state (set when the latch fires; cleared on reset()).
        self._latch_disables_rotation = bool(cfg.get("latch_disables_rotation", False))
        self._all_off = False

        # Shared scene params live in the rack `position` block (grid bounds,
        # map size, gripper bounds). Mode-specific position params come from the
        # ACTIVE block: rack=`position`, cabinet=`cabinet.position`, with fallback
        # to the rack block (via _pp) for anything the cabinet block omits.
        pcfg = dict(cfg.get("position", {}))
        ws_min = np.asarray(pcfg["workspace_bounds_min"], dtype=np.float32)
        ws_max = np.asarray(pcfg["workspace_bounds_max"], dtype=np.float32)
        map_size = int(pcfg.get("map_size", 100))

        if self._place_mode == "cabinet":
            # Point-attractor(s) toward the cabinet, plus an optional avoidance
            # blob (the closed cabinet to the right). Supports a SEQUENCE of
            # position stages (`cpos.stages`): the EE advances to the next target
            # when it arrives within that stage's arrival_radius_m (see
            # get_guidance); after the last stage it latches off. Lets us split
            # the placement into e.g. (1) approach in front → (2) onto the surface,
            # each tunable. Falls back to a single `target` when no stages given.
            cpos = dict(ccfg.get("position", {}))
            pos_params = cpos
            av_cfg = dict(cpos.get("avoidance", {}))
            avoidance_boxes = None
            if av_cfg.get("enabled", False) and av_cfg.get("box") is not None:
                b = av_cfg["box"]
                avoidance_boxes = [(np.asarray(b["center"], dtype=np.float32),
                                    np.asarray(b["size"], dtype=np.float32))]
            seed_extent = float(cpos.get("seed_extent_m", 0.04))
            av_weight = float(av_cfg.get("weight", 0.0))
            av_sigma = float(av_cfg.get("sigma", 3.0))
            stages_cfg = cpos.get("stages", None)
            # A stage target is either explicit coords (`target: [x,y,z]`) or a
            # workspace box referenced by name (`target_box: box_004`) → its center.
            boxes_lookup = None
            if stages_cfg and any(s.get("target_box") for s in stages_cfg):
                boxes_lookup = load_boxes(pcfg["boxes_path"])

            def _stage_target(s):
                if s.get("target_box"):
                    return np.asarray(
                        boxes_lookup[s["target_box"]]["center"], dtype=np.float32)
                return np.asarray(s["target"], dtype=np.float32)

            if stages_cfg:
                stage_specs = [(_stage_target(s),
                                float(s.get("arrival_radius_m", 0.08)))
                               for s in stages_cfg]
            else:
                stage_specs = [(np.asarray(cpos["target"], dtype=np.float32),
                                float(cpos.get("basin_latch_radius_m", 0.10)))]
            self._cab_stages = []
            for tgt, radius in stage_specs:
                vm_i = build_center_attractor_value_map(
                    tgt, ws_min=ws_min, ws_max=ws_max, map_size=map_size,
                    seed_extent_m=seed_extent, avoidance_boxes=avoidance_boxes,
                    avoidance_weight=av_weight, obstacle_sigma=av_sigma)
                self._cab_stages.append({
                    "vm": vm_i,
                    "grad": gradient_field_tensor(vm_i, self.device),
                    "target": tgt,
                    "radius": radius,
                })
            self._cab_stage_idx = 0
            vm = self._cab_stages[0]["vm"]
            self._stage_target_world = self._cab_stages[0]["target"]
        else:
            self._cab_stages = None
            pos_params = pcfg
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
            # Basin target — center of the rack's front-face affordance slab; the
            # DistanceScaler ramps the pull DOWN as the EE approaches it.
            self._stage_target_world = front_face_center(
                boxes[rack_name]["center"], boxes[rack_name]["size"],
                face_thickness_m=face_thickness_m,
                forward_extend_m=forward_extend_m,
                y_offset_m=basin_y_offset_m,
                z_offset_m=basin_z_offset_m,
            )

        def _pp(key, default):
            """Active-block position param with fallback to the shared rack block."""
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
        self._delta_norm_cap = float(_pp("delta_norm_cap", 0.02))

        # Basin latch — once the *measured* EE first comes within this radius of
        # the basin, the position branch turns OFF for the rest of the stage and
        # stays off (rotation steering keeps running). The basin is an APPROACH
        # point, not the final rack-slot pose; the DistanceScaler alone ramps the
        # pull to its floor near the basin but is non-latching, so if the EE
        # drifts back out the pull re-engages and the EE loops in/out of the
        # basin while the policy tries to align with the rack. The latch hands
        # final alignment entirely to the policy once we've arrived. Set to 0 to
        # disable (fall back to pure DistanceScaler behavior). Cleared on reset().
        self._basin_latch_radius_m = float(_pp("basin_latch_radius_m", 0.10))
        self._pos_latched = False

        # Log the first few pos_delta magnitudes so the user can see the field
        # actually nudging the trajectory (not destabilizing it). Tunable so we
        # can quiet it once the steering is tuned.
        self._diag_remaining = int(_pp("diag_log_calls", 30))

        logger.info(
            "CombinedBoxSteering: place_mode=%s mode=%s target=%s pos_strength=%s "
            "start_t=%s map_size=%d delta_norm_cap=%s latch=%s ws=[%s, %s]",
            self._place_mode,
            self.guidance_mode,
            np.array2string(np.asarray(self._stage_target_world), precision=3),
            _pp("guidance_strength", 0.05),
            _pp("start_guidance_timestep", 10_000),
            map_size,
            self._delta_norm_cap,
            self._basin_latch_radius_m,
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
        # Updates the live relative target_6d (R_base.T @ R_target_world).
        self._rot.set_current_gripper_rotation(ee_euler_xyz)
        if not self._rot_use_slerp:
            return
        # Decide this plan's long/short choice so the wrist keeps rotating about
        # one COMMITTED world-frame axis (prevents the replan/equator-crossing
        # direction flip). Compare the current short-way axis (grasp→target, in
        # world) to the committed axis: if they oppose, go the long way.
        R_base = _euler_to_matrix(ee_euler_xyz, self.device)        # (3, 3) world
        R_target = self._rot._R_target_world                        # (3, 3) world
        R_delta = R_target @ R_base.transpose(0, 1)                 # world grasp→target
        axis_w, angle = self._axis_angle(R_delta)
        if angle < 1e-3:
            self._rot_negate = False
            return
        if self._rot_commit_axis is None:
            # Commit at stage entry: reverse → opposite of the short-way axis.
            self._rot_commit_axis = (
                -axis_w if self._rot_reverse_direction else axis_w
            )
        if angle < self._rot_reverse_min_angle:
            # Close to target — always short way. Forcing the long way around a
            # small residual here would command a ~360° spin and fold the arm.
            self._rot_negate = False
        else:
            # Long way when the short-way axis opposes the committed direction.
            self._rot_negate = bool(np.dot(axis_w, self._rot_commit_axis) < 0.0)

    def reset(self) -> None:
        self._rot.reset()
        self._pos_latched = False
        self._all_off = False
        self._rot_commit_axis = None
        self._rot_negate = False
        # Rewind the cabinet position-stage sequence to the first target.
        if self._cab_stages is not None:
            self._cab_stage_idx = 0
            self._stage.value_map = self._cab_stages[0]["vm"]
            self._stage.gradient_field = self._cab_stages[0]["grad"]
            self._stage_target_world = self._cab_stages[0]["target"]

    # Lifecycle no-ops for run_experiment / policy compatibility.
    def setup_episode(self, task_name: str):
        return None, None

    def increment_step(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Guidance
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_angle(R: torch.Tensor) -> tuple[np.ndarray, float]:
        """Rotation matrix (3,3) -> (unit axis (3,), angle in [0, π])."""
        Rn = R.detach().cpu().numpy().astype(np.float64)
        cos = (np.trace(Rn) - 1.0) / 2.0
        angle = float(np.arccos(np.clip(cos, -1.0, 1.0)))
        ax = np.array([Rn[2, 1] - Rn[1, 2],
                       Rn[0, 2] - Rn[2, 0],
                       Rn[1, 0] - Rn[0, 1]], dtype=np.float64)
        n = np.linalg.norm(ax)
        axis = ax / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        return axis, angle

    @staticmethod
    def _slerp_from_identity_targets(
        target_6d: torch.Tensor,
        alphas: torch.Tensor,
        negate: bool,
        device,
    ) -> torch.Tensor:
        """Per-horizon SLERP targets from the RELATIVE identity to target_6d.

        Anchored at identity (the current EE pose in the relative frame), NOT the
        live prediction — so the targets are FIXED for the plan and don't flip as
        the denoiser's x0 estimate moves (that live-anchored flip was the
        oscillation bug). `negate=True` flips the target quaternion to the
        opposite hemisphere so the geodesic travels the LONG way around; the
        per-plan caller sets it to maintain the committed world-frame direction.

        target_6d: (6,); alphas: (H,). Returns (H, 6).
        """
        H = alphas.shape[0]
        q_id = torch.zeros(H, 4, device=device, dtype=alphas.dtype)
        q_id[:, 0] = 1.0  # identity quaternion (w=1)
        qt = matrix_to_quaternion(
            compute_rotation_matrix_from_ortho6d(target_6d.view(1, 6))
        ).squeeze(0)  # (4,)
        if float(qt[0]) < 0.0:
            qt = -qt                 # canonical short-way hemisphere from identity
        if negate:
            qt = -qt                 # flip to the long way
        qt = qt.view(1, 4).expand(H, 4).contiguous()

        dot = (q_id * qt).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        a = alphas.view(H, 1)
        safe = sin_theta.abs() > 1e-6
        w0 = torch.where(safe, torch.sin((1.0 - a) * theta) / sin_theta, 1.0 - a)
        w1 = torch.where(safe, torch.sin(a * theta) / sin_theta, a)
        q = w0 * q_id + w1 * qt
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return get_ortho6d_from_rotation_matrix(quaternion_to_matrix(q))  # (H, 6)

    def _rotation_slerp_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic (SLERP) rotation delta in [3:9].

        Mirrors TargetRotationSteering.get_guidance EXACTLY (same Tweedie x0,
        same dps/epsilon delta, same [3:9] placement) but replaces the single
        constant target with per-horizon SLERP targets along the SO(3) geodesic
        from the current pose to the relative target. Direction is committed in
        world frame (see set_current_gripper_rotation → `_rot_negate`), so it
        can't flip on replan. The per-horizon alpha ramps `rot_alpha_floor →
        rot_alpha_max` quadratically, so earlier waypoints take a smaller step
        along the geodesic — gradual onset like the linear branch's `ramp`.
        """
        rot = self._rot
        container = model_output if self.guidance_mode != "dps" else current_sample
        zero = torch.zeros_like(container)

        if rot._rotation_target_6d is None or rot.rotation_scheduler is None:
            return zero
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        if t > rot.start_guidance_timestep:
            return zero

        B, L, _ = model_output.shape
        H = min(self.horizon, L)

        abar = max(float(rot.rotation_scheduler.alphas_cumprod[t]), 1e-6)
        sqrt_abar = abar ** 0.5
        sqrt_1m = (1.0 - abar) ** 0.5

        eps_rot = model_output[:, :H, 3:9]
        x_t_rot = current_sample[:, :H, 3:9]
        x0_rot = (x_t_rot - sqrt_1m * eps_rot) / sqrt_abar  # (B, H, 6)

        target_single = rot._rotation_target_6d.to(container.device).view(6)
        h_idx = torch.arange(H, device=container.device, dtype=x0_rot.dtype)
        alphas = self._rot_alpha_floor + (
            self._rot_alpha_max - self._rot_alpha_floor
        ) * (h_idx / max(H - 1, 1)) ** 2  # (H,)

        # Per-horizon geodesic targets, anchored at the current pose (identity in
        # the relative frame) — fixed for the plan, so they don't oscillate as
        # x0_rot moves. `_rot_negate` (decided per plan to hold the committed
        # world axis) routes the SLERP the long way when needed.
        slerp_target = self._slerp_from_identity_targets(
            target_single, alphas, negate=self._rot_negate, device=container.device
        ).unsqueeze(0)  # (1, H, 6) — broadcasts over batch

        if self.guidance_mode == "dps":
            # Nudge x_{t-1} toward the per-horizon geodesic target (same sign as
            # TargetRotationSteering's proven dps path).
            delta = self.guidance_strength_rot * (slerp_target - x0_rot)
        else:
            coeff = sqrt_1m / sqrt_abar
            delta = self.guidance_strength_rot * coeff * (x0_rot - slerp_target)

        zero[:, :H, 3:9] = delta
        return zero

    def get_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        obs_embedding: Any,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Rotation delta in [3:9] + value-map position delta in [0:3]."""
        # ALL steering off (basin reached with latch_disables_rotation) → zeros.
        if self._all_off:
            container = model_output if self.guidance_mode != "dps" else current_sample
            return torch.zeros_like(container)
        # Rotation branch returns a full-size tensor (delta only in [3:9]).
        if self._rot_use_slerp:
            guidance = self._rotation_slerp_guidance(
                current_sample, timestep, model_output
            )
        else:
            guidance = self._rot.get_guidance(
                current_sample, timestep, obs_embedding, model_output
            )

        # Position-stage progression / basin latch. The measured EE is fixed
        # across this plan's denoising loop (it's the observation pose), so these
        # checks are effectively per-plan.
        if self._cab_stages is not None:
            # Cabinet: advance through the position-stage sequence as the EE
            # arrives at each target; latch off after the last. One-way (monotonic).
            ee = self._coords.current_gripper_pos
            if not self._pos_latched and ee is not None:
                cur = self._cab_stages[self._cab_stage_idx]
                d = float(torch.norm(
                    ee - torch.as_tensor(cur["target"], dtype=ee.dtype, device=ee.device)
                ).item())
                if cur["radius"] > 0.0 and d <= cur["radius"]:
                    if self._cab_stage_idx + 1 < len(self._cab_stages):
                        self._cab_stage_idx += 1
                        nxt = self._cab_stages[self._cab_stage_idx]
                        self._stage.value_map = nxt["vm"]
                        self._stage.gradient_field = nxt["grad"]
                        self._stage_target_world = nxt["target"]
                        logger.info(
                            "CombinedBoxSteering: position stage %d reached "
                            "(d=%.3fm) → advancing to stage %d target=%s",
                            self._cab_stage_idx, d, self._cab_stage_idx + 1,
                            np.array2string(nxt["target"], precision=3))
                    else:
                        self._pos_latched = True
                        self._all_off = self._latch_disables_rotation
                        logger.info(
                            "CombinedBoxSteering: final position stage reached "
                            "(d=%.3fm) — %s steering OFF.", d,
                            "ALL" if self._all_off else "position")
            if self._all_off:
                return torch.zeros_like(guidance)
            if self._pos_latched:
                return guidance
        elif self._basin_latch_radius_m > 0.0:
            # Rack (single target): latch off once within the radius.
            ee = self._coords.current_gripper_pos
            if not self._pos_latched and ee is not None:
                basin = torch.as_tensor(
                    self._stage_target_world, dtype=ee.dtype, device=ee.device)
                d = float(torch.norm(ee - basin).item())
                if d <= self._basin_latch_radius_m:
                    self._pos_latched = True
                    self._all_off = self._latch_disables_rotation
                    logger.info(
                        "CombinedBoxSteering: basin latch ENGAGED (d=%.3fm <= "
                        "%.3fm) — %s steering OFF for the rest of the stage.",
                        d, self._basin_latch_radius_m,
                        "ALL" if self._all_off else "position")
            if self._all_off:
                return torch.zeros_like(guidance)
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
