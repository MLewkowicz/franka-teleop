"""Deploy a trained 3D Diffuser Actor checkpoint on the real Franka.

Chunked playback: an inference worker grabs two ZED frames + the current
end-effector pose when the executor asks for a new plan, builds a
`core.types.Observation`, and calls `policy.forward(obs)` to produce an
absolute trajectory. The main loop executes that trajectory to completion
before requesting another one. If execution stalls past a timeout, the executor
abandons the old plan and pauses while the worker generates a replacement. The
gripper command is issued when the executed trajectory step flips relative to
what we last commanded.

Stage transitions (grasp → place → done) are driven by SpaceMouse buttons:
    LEFT  short tap   → toggle ENABLED (closed-loop control on/off)
    RIGHT short tap   → advance stage (grasp → place → exit)

Launch:
    uv run python deploy_diffuser_actor.py \\
        deploy.policy_config=/path/to/policy.yaml \\
        deploy.langsteer_path=$HOME/Documents/michal/LangSteer
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
import os
import sys
import threading
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("deploy_diffuser_actor")


@dataclass
class InferencePlan:
    sequence: int
    stage_idx: int
    epoch: int
    created_at: float
    obs_started_at: float
    trajectory: np.ndarray
    gripper: np.ndarray


class LatestPlanSlot:
    """Single-slot handoff from the inference worker to the executor loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._plan: InferencePlan | None = None

    def publish(self, plan: InferencePlan) -> None:
        with self._lock:
            self._plan = plan

    def latest_after(self, sequence: int) -> InferencePlan | None:
        with self._lock:
            if self._plan is None or self._plan.sequence <= sequence:
                return None
            return self._plan


# ---------------------------------------------------------------------------
# Sys.path wiring — LangSteer is installed on the robot machine and provides
# the policy/model code; we add it to PYTHONPATH before importing the policy.
# ---------------------------------------------------------------------------

def _wire_langsteer(langsteer_path: str) -> None:
    p = Path(langsteer_path).expanduser().resolve()
    if not (p / "policies" / "diffuser_actor.py").is_file():
        raise RuntimeError(
            f"deploy.langsteer_path={p} does not look like a LangSteer checkout "
            "(missing policies/diffuser_actor.py). Set it to your LangSteer "
            "repo root."
        )
    sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Policy instantiation
# ---------------------------------------------------------------------------

def _build_policy(deploy_cfg: DictConfig):
    """Load the policy yaml, build the right variant, load the checkpoint."""
    from policies.diffuser_actor import build_diffuser_actor_policy

    policy_cfg = OmegaConf.load(deploy_cfg.policy_config)

    checkpoint = policy_cfg.get("ckpt_path")

    # The factory reads use_primitive_id / use_object_id from cfg, so make
    # sure the policy yaml has them.
    policy = build_diffuser_actor_policy(policy_cfg)
    policy.load_checkpoint(checkpoint)
    policy.reset()
    return policy, bool(policy_cfg.get("relative", False))


# ---------------------------------------------------------------------------
# Camera + extrinsics setup
# ---------------------------------------------------------------------------

def _setup_cameras(cfg: DictConfig):
    from clear_franka.camera import make_zed_camera, enabled_camera_names
    from clear_franka.diffuser_actor_io import (
        CameraPreprocessor, load_extrinsics_json,
    )

    names = enabled_camera_names(cfg)
    if "hand" not in names or "third_person" not in names:
        raise RuntimeError(
            f"deploy needs both 'hand' and 'third_person' enabled in cameras.* "
            f"of conf/config.yaml; got: {names}"
        )

    cam_hand = make_zed_camera(cfg, "hand")
    cam_tp = make_zed_camera(cfg, "third_person")

    hand_ext = load_extrinsics_json(cfg.cameras.hand.extrinsics_path)
    tp_ext = load_extrinsics_json(cfg.cameras.third_person.extrinsics_path)
    pre_hand = CameraPreprocessor(hand_ext, mount="hand")
    pre_tp = CameraPreprocessor(tp_ext, mount="third_person")
    return cam_hand, cam_tp, pre_hand, pre_tp


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_observation(rgb_tp_200, pcd_tp_200, rgb_hand_200, pcd_hand_200,
                       ee_pos, ee_rot, gripper_command):
    """Pack into the core.types.Observation the DiffuserActor expects.

    Keys match the policy's cfg.cameras = ["front", "wrist"]; values:
        rgb["front"], rgb["wrist"]  — (200, 200, 3) uint8 RGB
        depth["front"], depth["wrist"]  — (200, 200, 3) float32 base-frame XYZ
        ee_pose  — concat(xyz, euler_XYZ, gripper)   shape (7,)
    """
    from core.types import Observation
    from clear_franka.diffuser_actor_io import ee_rot_to_euler_xyz

    euler = ee_rot_to_euler_xyz(ee_rot)  # (3,) pytorch3d "XYZ" intrinsic
    ee_pose = np.concatenate([
        np.asarray(ee_pos, dtype=np.float32),
        euler,
        np.array([float(gripper_command)], dtype=np.float32),
    ])

    return Observation(
        rgb={"front": rgb_tp_200, "wrist": rgb_hand_200},
        depth={"front": pcd_tp_200, "wrist": pcd_hand_200},
        proprio=np.zeros(0, dtype=np.float32),  # unused by DiffuserActor
        ee_pose=ee_pose,
        instruction="",                          # unused in primitive+object mode
    )


def _read_ee_pose_from_state(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    O_T_EE = np.asarray(state["O_T_EE"], dtype=np.float64).reshape(4, 4)
    ee_pos = O_T_EE[:3, 3].copy()
    ee_rot = O_T_EE[:3, :3].copy()
    return ee_pos, ee_rot, O_T_EE


def _make_T_gripper_to_base(ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
    T_g2b = np.eye(4)
    T_g2b[:3, :3] = ee_rot
    T_g2b[:3, 3] = ee_pos
    return T_g2b


def _update_visualizer_robot_state(visualizer, state: dict | None) -> None:
    if visualizer is None or state is None:
        return

    joint_pos = np.asarray(state["q"], dtype=float) if "q" in state else None
    visualizer.update(joint_pos)
    if "O_T_EE" in state:
        visualizer.update_eef_frame(np.asarray(state["O_T_EE"], dtype=float).reshape(4, 4))


def _extract_gripper_plan(action, horizon: int) -> np.ndarray:
    trajectory = np.asarray(action.trajectory)
    if trajectory.ndim == 2 and trajectory.shape[1] >= 7:
        return trajectory[:horizon, 6].astype(np.float64)
    return np.full(horizon, float(action.gripper), dtype=np.float64)


def _plan_start_index(
    plan: InferencePlan,
    current_ee_pos: np.ndarray,
    plan_dt: float,
    now: float,
) -> int:
    horizon = len(plan.trajectory)
    if horizon <= 1:
        return 0

    distances = np.linalg.norm(plan.trajectory[:, :3] - current_ee_pos[None, :], axis=1)
    closest_next = int(np.argmin(distances)) + 1
    latency_skip = int(max(0.0, now - plan.obs_started_at) / plan_dt)
    return min(max(1, closest_next, latency_skip), horizon - 1)


def _make_cartesian_trajectory_for_plan(
    plan: InferencePlan,
    start_index: int,
    plan_dt: float,
    euler_to_matrix_fn,
    *,
    max_linear_vel: float,
    max_angular_vel: float,
    min_segment_dt: float,
):
    from clear_franka.cartesian_trajectory import CartesianTrajectory

    suffix = plan.trajectory[start_index:, :6]
    if len(suffix) == 1:
        suffix = np.vstack([suffix, suffix])
    times = np.arange(len(suffix), dtype=np.float64) * float(plan_dt)
    trajectory = CartesianTrajectory.from_euler_xyz(
        suffix,
        times,
        euler_to_matrix_fn=euler_to_matrix_fn,
        smooth_orientation=True,
    )
    return trajectory.retime(
        max_linear_vel=max_linear_vel,
        max_angular_vel=max_angular_vel,
        min_segment_dt=min_segment_dt,
    )


def _sample_cartesian_trajectory_positions(
    trajectory,
    dt: float,
    max_samples: int = 500,
) -> np.ndarray:
    duration = float(trajectory.duration)
    if duration <= 0.0:
        position, _rotation = trajectory.interpolate(0.0)
        return position.reshape(1, 3)

    num_samples = max(int(np.ceil(duration / float(dt))) + 1, 2)
    num_samples = min(num_samples, int(max_samples))
    times = np.linspace(0.0, duration, num_samples)
    return np.stack([trajectory.interpolate(t)[0] for t in times], axis=0)


def _start_inference_worker(
    *,
    policy,
    policy_lock: threading.Lock,
    robot,
    cam_hand,
    cam_tp,
    pre_hand,
    pre_tp,
    latest_plan: LatestPlanSlot,
    enabled_event: threading.Event,
    request_event: threading.Event,
    stop_event: threading.Event,
    stage_state: dict[str, int],
) -> threading.Thread:
    def worker() -> None:
        sequence = 0
        while not stop_event.is_set():
            if not enabled_event.wait(0.05):
                continue
            if not request_event.wait(0.05):
                continue
            request_event.clear()
            if not enabled_event.is_set():
                continue

            try:
                obs_started_at = time.monotonic()
                state = robot.wait_for_state(timeout=1.0)
                ee_pos, ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                T_g2b = _make_T_gripper_to_base(ee_pos, ee_rot)

                hand_frame = cam_hand.grab_frame()
                tp_frame = cam_tp.grab_frame()
                if hand_frame is None or tp_frame is None:
                    logger.warning("Camera grab failed — skipping inference")
                    request_event.set()
                    continue
                rgb_hand_full, depth_hand_full = hand_frame
                rgb_tp_full, depth_tp_full = tp_frame

                rgb_hand_200, pcd_hand_200 = pre_hand.process(
                    rgb_hand_full, depth_hand_full, T_g2b
                )
                rgb_tp_200, pcd_tp_200 = pre_tp.process(
                    rgb_tp_full, depth_tp_full
                )

                obs = _build_observation(
                    rgb_tp_200, pcd_tp_200,
                    rgb_hand_200, pcd_hand_200,
                    ee_pos, ee_rot, stage_state["gripper_cmd"],
                )
                forward_started_at = time.monotonic()
                with policy_lock:
                    stage_idx = stage_state["idx"]
                    epoch = stage_state["epoch"]
                    action = policy.forward(obs)
                forward_s = time.monotonic() - forward_started_at

                # DiffuserActorBasePolicy.forward() already converts relative
                # model outputs into absolute poses before returning Action.
                trajectory = np.asarray(action.trajectory, dtype=np.float64).copy()
                horizon = trajectory.shape[0]
                if not enabled_event.is_set():
                    continue
                plan = InferencePlan(
                    sequence=sequence,
                    stage_idx=stage_idx,
                    epoch=epoch,
                    created_at=time.monotonic(),
                    obs_started_at=obs_started_at,
                    trajectory=trajectory,
                    gripper=_extract_gripper_plan(action, horizon),
                )
                latest_plan.publish(plan)
                logger.info(
                    "  published plan %d horizon=%d forward=%.2fs total=%.2fs first=%s last=%s",
                    plan.sequence,
                    horizon,
                    forward_s,
                    plan.created_at - obs_started_at,
                    np.array2string(plan.trajectory[0, :3], precision=3),
                    np.array2string(plan.trajectory[-1, :3], precision=3),
                )
                sequence += 1
            except Exception as e:
                print(e)
                logger.exception("Inference worker failed")
                stop_event.set()
                break

    thread = threading.Thread(target=worker, name="diffuser-inference", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    _wire_langsteer(cfg.deploy.langsteer_path)

    # Late imports — zero_franky and the policy depend on host-side venvs.
    from zero_franky import Robot, setup_zero_franky
    from franky import Affine, JointMotion, JointState
    from clear_franka.utils import LoopRatePrinter
    from clear_franka.diffuser_actor_io import euler_xyz_to_matrix
    from clear_franka.geometry import pack_Rp
    from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
    from clear_franka.visualization import CortadoViserVisualizer
    from threed_mouse import ThreeDMouse

    from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS

    # ----- policy -----
    policy, policy_relative = _build_policy(cfg.deploy)

    # Stage 0: grasp, Stage 1: place — primitive ids in the trained vocab.
    stages = [
        {"primitive": 0, "object": 0, "label": "grasp glass"},
        {"primitive": 1, "object": 0, "label": "place glass"},
    ]
    stage_idx = 0
    policy.set_primitive(stages[stage_idx]["primitive"])
    policy.set_object(stages[stage_idx]["object"])
    logger.info(f"[stage 0] {stages[stage_idx]['label']}")

    # ----- cameras -----
    # make_zed_camera() returns an already-opened ZedCamera (it calls
    # zed.open() inside __init__). We use synchronous grab_frame() per tick,
    # so we deliberately do NOT call .run() — that would start a background
    # capture thread and grab_frame() warns it must not run concurrently with
    # it (see camera.py:237 docstring).
    cam_hand, cam_tp, pre_hand, pre_tp = _setup_cameras(cfg)

    # ----- gripper (init pattern mirrors teleop.py:184-200) -----
    gc = cfg.gripper
    gripper = None
    if gc.get("enabled", False):
        gripper = RobotiqGripperProxy(
            server_host=gc.host,
            server_port=int(gc.port),
            com_port=gc.com_port,
            device_id=int(gc.device_id),
            connection_type=gc.connection_type,
            tcp_host=gc.tcp_host,
            tcp_port=int(gc.tcp_port),
            auto_activate=True,
        )
        # Default open at start (matches training: episodes begin with gripper open).
        gripper.move_width(gc.open_width_m, wait=False)

    # ----- visualization -----
    vc = cfg.get("visualization", {})
    visualizer = CortadoViserVisualizer(
        host=vc.get("host", "0.0.0.0"),
        port=int(vc.get("port", 8080)),
    )
    visualizer.update_gripper_width(
        cfg.gripper.open_width_m,
        max_width_m=cfg.gripper.max_width_m,
    )

    # ----- robot -----
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port,
                      pub_port=cfg.zero_franky.pub_port)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    reset_joint_config = np.asarray(cfg.teleop.reset_joint_config, dtype=float)
    logger.info(f"Resetting to start config {reset_joint_config}")
    robot.move(JointMotion(JointState(reset_joint_config),
                            relative_dynamics_factor=0.1),
               asynchronous=False)

    mouse = None
    try:
        mouse = ThreeDMouse(control_rate=cfg.teleop.spacemouse.control_rate)
        mouse.run()
        logger.info("SpaceMouse: LEFT toggle ENABLE, RIGHT advance stage")
    except Exception as e:
        logger.warning(f"No SpaceMouse ({e}); buttons disabled — Ctrl-C to stop.")

    enabled = False
    rate = LoopRatePrinter()
    plan_hz = float(cfg.deploy.get("control_hz", 10.0))
    execution_hz = float(cfg.deploy.get("execution_hz", 100.0))
    plan_dt = 1.0 / plan_hz
    execution_dt = 1.0 / execution_hz
    policy_lock = threading.Lock()
    latest_plan = LatestPlanSlot()
    enabled_event = threading.Event()
    request_event = threading.Event()
    stop_event = threading.Event()
    stage_state: dict = {"idx": stage_idx, "epoch": 0, "gripper_cmd": 1.0}

    with contextlib.ExitStack() as stack:
        stack.enter_context(cam_hand)
        stack.enter_context(cam_tp)
        if gripper is not None:
            stack.enter_context(gripper)
        if mouse is not None:
            stack.callback(mouse.close)
        stack.callback(rate.newline)
        tracker = stack.enter_context(robot.start_cartesian_impedance_session(
            period=0.001,
            translational_stiffness=cfg.deploy.translational_stiffness,
            rotational_stiffness=cfg.deploy.rotational_stiffness,
            nullspace_stiffness=cfg.deploy.nullspace_stiffness,
            lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
            upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        ))
        robot.start_state_stream(timeout_ms=250)
        stack.callback(robot.stop_state_stream)
        inference_thread = _start_inference_worker(
            policy=policy,
            policy_lock=policy_lock,
            robot=robot,
            cam_hand=cam_hand,
            cam_tp=cam_tp,
            pre_hand=pre_hand,
            pre_tp=pre_tp,
            latest_plan=latest_plan,
            enabled_event=enabled_event,
            request_event=request_event,
            stop_event=stop_event,
            stage_state=stage_state,
        )
        stack.callback(lambda: (stop_event.set(), enabled_event.set(), inference_thread.join(timeout=1.0)))

        prev_left = 0
        prev_right = 0
        active_plan: InferencePlan | None = None
        active_cartesian_trajectory = None
        active_plan_index_offset = 0
        active_index = 0
        active_plan_started_at = 0.0
        consumed_sequence = -1
        waiting_for_plan = False
        next_tick = time.monotonic()
        last_viz_update = 0.0
        viz_dt = 1.0 / 5.0
        plan_timeout_s = 10.0
        plan_completion_tolerance_m = 0.015
        plan_hard_skip_m = 0.12
        plan_max_linear_vel_m_s = 0.10
        plan_max_angular_vel_rad_s = 0.75
        plan_timeout_grace_s = 2.0

        while not stop_event.is_set():
            rate.start_tick()
            # ---------- button polling ----------
            if mouse is not None:
                sample = mouse.get_controller_state()
                if sample is not None:
                    buttons = np.asarray(sample.buttons, dtype=int)
                    left = int(buttons[0]) if len(buttons) > 0 else 0
                    right = int(buttons[1]) if len(buttons) > 1 else 0
                    if left and not prev_left:
                        enabled = not enabled
                        active_plan = None
                        active_cartesian_trajectory = None
                        active_plan_index_offset = 0
                        active_plan_started_at = 0.0
                        waiting_for_plan = False
                        visualizer.clear_plan_waypoints()
                        stage_state["epoch"] += 1
                        if enabled:
                            enabled_event.set()
                            request_event.set()
                            waiting_for_plan = True
                        else:
                            enabled_event.clear()
                            request_event.clear()
                        logger.info(f"  {'ENABLED' if enabled else 'DISABLED'}")
                    if right and not prev_right:
                        stage_idx += 1
                        if stage_idx >= len(stages):
                            logger.info("  All stages done — exiting.")
                            break
                        enabled_event.clear()
                        request_event.clear()
                        active_plan = None
                        active_cartesian_trajectory = None
                        active_plan_index_offset = 0
                        active_plan_started_at = 0.0
                        waiting_for_plan = False
                        visualizer.clear_plan_waypoints()
                        with policy_lock:
                            policy.set_primitive(stages[stage_idx]["primitive"])
                            policy.set_object(stages[stage_idx]["object"])
                            policy.reset()
                            stage_state["idx"] = stage_idx
                            stage_state["epoch"] += 1
                        if enabled:
                            enabled_event.set()
                            request_event.set()
                            waiting_for_plan = True
                        logger.info(f"[stage {stage_idx}] {stages[stage_idx]['label']}")
                    prev_left = left
                    prev_right = right

            now = time.monotonic()
            if now - last_viz_update >= viz_dt:
                _update_visualizer_robot_state(visualizer, robot.latest_state)
                last_viz_update = now

            if not enabled:
                rate.finish_tick()
                next_tick += execution_dt
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
                continue

            if active_plan is None and not waiting_for_plan:
                request_event.set()
                waiting_for_plan = True

            # ---------- adopt latest completed plan ----------
            current_epoch = stage_state["epoch"]
            plan = latest_plan.latest_after(consumed_sequence)
            if plan is not None and active_plan is None:
                consumed_sequence = plan.sequence
                if plan.stage_idx == stage_idx and plan.epoch == current_epoch:
                    state = robot.latest_state
                    if state is None:
                        state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                    ee_pos, _ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                    min_dist = float(np.min(
                        np.linalg.norm(plan.trajectory[:, :3] - ee_pos[None, :], axis=1)
                    ))
                    if min_dist > plan_hard_skip_m:
                        visualizer.update_plan_waypoints(plan.trajectory, 0)
                        logger.info(
                            "  skipping plan %d: nearest waypoint %.1f cm from EE; "
                            "ee=%s first=%s last=%s",
                            plan.sequence,
                            min_dist * 100.0,
                            np.array2string(ee_pos, precision=3),
                            np.array2string(plan.trajectory[0, :3], precision=3),
                            np.array2string(plan.trajectory[-1, :3], precision=3),
                        )
                        request_event.set()
                    else:
                        active_plan = plan
                        waiting_for_plan = False
                        adopted_at = time.monotonic()
                        active_index = _plan_start_index(
                            plan,
                            ee_pos,
                            plan_dt,
                            adopted_at,
                        )
                        active_plan_index_offset = active_index
                        active_cartesian_trajectory = _make_cartesian_trajectory_for_plan(
                            plan,
                            active_index,
                            plan_dt,
                            euler_xyz_to_matrix,
                            max_linear_vel=plan_max_linear_vel_m_s,
                            max_angular_vel=plan_max_angular_vel_rad_s,
                            min_segment_dt=execution_dt,
                        )
                        active_plan_started_at = adopted_at
                        visualizer.update_plan_waypoints(plan.trajectory, active_index)
                        visualizer.update_interpolated_plan_path(
                            _sample_cartesian_trajectory_positions(
                                active_cartesian_trajectory,
                                execution_dt,
                            )
                        )
                        logger.info(
                            "  adopted plan %d idx=%d/%d age=%.0fms infer=%.0fms",
                            plan.sequence,
                            active_index,
                            len(plan.trajectory),
                            (time.monotonic() - plan.created_at) * 1000.0,
                            (plan.created_at - plan.obs_started_at) * 1000.0,
                        )
                else:
                    request_event.set()

            # ---------- execute the active plan suffix ----------
            if active_plan is not None and active_index < len(active_plan.trajectory):
                state = robot.latest_state
                if state is None:
                    state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                ee_pos, _ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                assert active_cartesian_trajectory is not None

                elapsed = time.monotonic() - active_plan_started_at
                local_index = active_cartesian_trajectory.waypoint_index_at(elapsed)
                active_index = min(
                    active_plan_index_offset + local_index,
                    len(active_plan.trajectory) - 1,
                )
                target_xyz, target_rot = active_cartesian_trajectory.interpolate(elapsed)
                visualizer.update_plan_waypoints(active_plan.trajectory, active_index)

                # Optional safety clip — keep targets inside the recorded workspace.
                lo = np.array(cfg.deploy.workspace_lo, dtype=np.float64)
                hi = np.array(cfg.deploy.workspace_hi, dtype=np.float64)
                target_xyz = np.clip(target_xyz, lo, hi)

                tracker.set_cartesian_reference(Affine(pack_Rp(target_rot, target_xyz)))

                cmd_state = 1.0 if active_plan.gripper[active_index] >= 0.5 else 0.0
                if gripper is not None and cmd_state != stage_state["gripper_cmd"]:
                    width = (cfg.gripper.open_width_m if cmd_state == 1.0
                             else cfg.gripper.close_width_m)
                    logger.info(f"  gripper → {'OPEN' if cmd_state == 1.0 else 'CLOSE'}")
                    gripper.move_width(width, wait=False)
                    visualizer.update_gripper_width(width, max_width_m=cfg.gripper.max_width_m)
                    stage_state["gripper_cmd"] = cmd_state

                final_dist = float(np.linalg.norm(
                    active_plan.trajectory[-1, :3] - ee_pos
                ))
                active_plan_timeout_s = max(
                    plan_timeout_s,
                    active_cartesian_trajectory.duration + plan_timeout_grace_s,
                )
                if (
                    active_index >= len(active_plan.trajectory) - 1
                    and final_dist <= plan_completion_tolerance_m
                ):
                    logger.info("  completed plan %d", active_plan.sequence)
                    visualizer.clear_plan_waypoints()
                    active_plan = None
                    active_cartesian_trajectory = None
                    active_plan_index_offset = 0
                    active_plan_started_at = 0.0
                    request_event.set()
                    waiting_for_plan = True
                elif time.monotonic() - active_plan_started_at > active_plan_timeout_s:
                    logger.info(
                        "  plan %d timed out after %.1fs at idx=%d/%d; requesting replacement",
                        active_plan.sequence,
                        active_plan_timeout_s,
                        active_index,
                        len(active_plan.trajectory),
                    )
                    visualizer.clear_plan_waypoints()
                    active_plan = None
                    active_cartesian_trajectory = None
                    active_plan_index_offset = 0
                    active_plan_started_at = 0.0
                    request_event.set()
                    waiting_for_plan = True

            elif active_plan is not None:
                visualizer.clear_plan_waypoints()
                active_plan = None
                active_cartesian_trajectory = None
                active_plan_index_offset = 0

            rate.finish_tick()
            next_tick += execution_dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    return 0


if __name__ == "__main__":
    main()
