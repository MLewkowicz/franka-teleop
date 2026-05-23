"""Deploy a trained 3D Diffuser Actor checkpoint on the real Franka.

MPC-style: every tick, grab two ZED frames + the current end-effector pose,
build a `core.types.Observation`, call `policy.forward(obs)` which returns a
20-step absolute trajectory, send only the *first* pose to the Cartesian
impedance controller, then loop. The gripper command is issued whenever the
policy's predicted gripper bit flips relative to what we last commanded.

Stage transitions (grasp → place → done) are driven by SpaceMouse buttons:
    LEFT  short tap   → toggle ENABLED (closed-loop control on/off)
    RIGHT short tap   → advance stage (grasp → place → exit)
    LEFT  long press  → reset to start joint config (same as teleop)

Launch:
    uv run python deploy_diffuser_actor.py \\
        deploy.checkpoint=/path/to/last.pth \\
        deploy.policy_config=/path/to/policy.yaml \\
        deploy.extrinsics_hand=./data/extrinsics_hand.json \\
        deploy.extrinsics_third_person=./data/extrinsics_third_person.json \\
        deploy.langsteer_path=$HOME/Documents/michal/LangSteer
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("deploy_diffuser_actor")


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

    # Allow overrides on the CLI (e.g. deploy.policy.embedding_dim=192).
    if "policy_overrides" in deploy_cfg and deploy_cfg.policy_overrides:
        policy_cfg = OmegaConf.merge(policy_cfg,
                                     OmegaConf.create(deploy_cfg.policy_overrides))

    # The factory reads use_primitive_id / use_object_id from cfg, so make
    # sure the policy yaml has them.
    policy = build_diffuser_actor_policy(policy_cfg)
    policy.load_checkpoint(deploy_cfg.checkpoint)
    policy.reset()
    return policy


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

    hand_ext = load_extrinsics_json(cfg.deploy.extrinsics_hand)
    tp_ext = load_extrinsics_json(cfg.deploy.extrinsics_third_person)
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if "deploy" not in cfg:
        raise RuntimeError(
            "conf/config.yaml is missing the `deploy:` section. Add it (see "
            "conf/deploy_example.yaml in this branch) or pass deploy.* overrides "
            "on the CLI."
        )

    _wire_langsteer(cfg.deploy.langsteer_path)

    # Late imports — zero_franky and the policy depend on host-side venvs.
    from zero_franky import Robot, setup_zero_franky
    from zero_franky.tracker_policies import passthrough_cartesian
    from franky import Affine, JointMotion, JointState
    from clear_franka.utils import LoopRatePrinter
    from clear_franka.diffuser_actor_io import euler_xyz_to_matrix
    from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
    from threed_mouse import ThreeDMouse

    from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS

    # ----- policy -----
    policy = _build_policy(cfg.deploy)

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
    cam_hand, cam_tp, pre_hand, pre_tp = _setup_cameras(cfg)
    cam_hand.start()
    cam_tp.start()

    # ----- gripper (init pattern mirrors teleop.py:184-200) -----
    gc = cfg.gripper
    gripper = None
    if gc.get("enabled", False):
        gripper = RobotiqGripperProxy(
            server_host=gc.get("host", cfg.zero_franky.ip),
            server_port=int(gc.get("port", cfg.zero_franky.port)),
            com_port=gc.get("com_port", "auto"),
            device_id=int(gc.get("device_id", 9)),
            connection_type=gc.get("connection_type", "RTU"),
            tcp_host=gc.get("tcp_host", "127.0.0.1"),
            tcp_port=int(gc.get("tcp_port", 54321)),
            auto_activate=bool(gc.get("activate_on_start", True)),
        )
        # Default open at start (matches training: episodes begin with gripper open).
        gripper.move_width(gc.open_width_m, wait=False)
    gripper_command_state = 1.0  # 1 = open, 0 = closed (matches training format)

    # ----- robot -----
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port,
                      pub_port=cfg.zero_franky.pub_port)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    reset_joint_config = np.asarray(cfg.teleop.reset_joint_config, dtype=float)
    logger.info(f"Resetting to start config {reset_joint_config}")
    robot.move(JointMotion(JointState(reset_joint_config),
                            dynamics_factor=0.2,
                            asynchronous=False))

    # SpaceMouse for the two buttons.
    mouse = None
    try:
        mouse = ThreeDMouse(control_rate=cfg.teleop.spacemouse.control_rate)
        mouse.run()
        logger.info("SpaceMouse: LEFT toggle ENABLE, RIGHT advance stage")
    except Exception as e:
        logger.warning(f"No SpaceMouse ({e}); buttons disabled — Ctrl-C to stop.")

    enabled = False
    rate = LoopRatePrinter()

    tracker = robot.start_cartesian_impedance_session(
        passthrough_cartesian,
        period=cfg.teleop.period,
        translational_stiffness=cfg.teleop.translational_stiffness,
        rotational_stiffness=cfg.teleop.rotational_stiffness,
        nullspace_stiffness=cfg.teleop.nullspace_stiffness,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
    )
    try:
        prev_left = 0
        prev_right = 0
        forward_count = 0

        while True:
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
                        logger.info(f"  {'ENABLED' if enabled else 'DISABLED'}")
                    if right and not prev_right:
                        stage_idx += 1
                        if stage_idx >= len(stages):
                            logger.info("  All stages done — exiting.")
                            break
                        policy.set_primitive(stages[stage_idx]["primitive"])
                        policy.set_object(stages[stage_idx]["object"])
                        policy.reset()  # clear gripper history at the boundary
                        logger.info(f"[stage {stage_idx}] {stages[stage_idx]['label']}")
                    prev_left = left
                    prev_right = right

            if not enabled:
                rate.finish_tick()
                continue

            # ---------- read robot state ----------
            teleop_state = robot.get_last_teleop_state()
            O_T_EE = np.asarray(teleop_state["O_T_EE"], dtype=np.float64).reshape(4, 4)
            ee_pos = O_T_EE[:3, 3].copy()
            ee_rot = O_T_EE[:3, :3].copy()
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = ee_rot
            T_g2b[:3, 3] = ee_pos

            # ---------- grab cameras ----------
            hand_frame = cam_hand.grab_frame()
            tp_frame = cam_tp.grab_frame()
            if hand_frame is None or tp_frame is None:
                logger.warning("Camera grab failed — skipping tick")
                rate.finish_tick()
                continue
            rgb_hand_full, depth_hand_full = hand_frame
            rgb_tp_full, depth_tp_full = tp_frame

            rgb_hand_200, pcd_hand_200 = pre_hand.process(
                rgb_hand_full, depth_hand_full, T_g2b
            )
            rgb_tp_200, pcd_tp_200 = pre_tp.process(
                rgb_tp_full, depth_tp_full
            )

            # ---------- build obs + forward ----------
            obs = _build_observation(
                rgb_tp_200, pcd_tp_200,
                rgb_hand_200, pcd_hand_200,
                ee_pos, ee_rot, gripper_command_state,
            )
            t0 = time.perf_counter()
            action = policy.forward(obs)
            forward_ms = (time.perf_counter() - t0) * 1000.0
            forward_count += 1
            if forward_count <= 5:
                logger.info(
                    f"  forward[{forward_count}] {forward_ms:.1f}ms  "
                    f"traj[0]={action.trajectory[0]}  gripper={action.gripper:.2f}"
                )

            # ---------- execute first pose only (MPC) ----------
            target_xyz = action.trajectory[0, :3].astype(np.float64)
            target_euler = action.trajectory[0, 3:6].astype(np.float64)
            target_rot = euler_xyz_to_matrix(target_euler)

            # Optional safety clip — keep targets inside the recorded workspace.
            lo = np.array(cfg.deploy.workspace_lo, dtype=np.float64)
            hi = np.array(cfg.deploy.workspace_hi, dtype=np.float64)
            target_xyz = np.clip(target_xyz, lo, hi)

            try:
                tracker.set_cartesian_reference(Affine(target_xyz, target_rot))
            except Exception as exc:
                logger.error(f"  set_cartesian_reference failed: {exc}")
                enabled = False

            # ---------- gripper command on state change ----------
            cmd_state = 1.0 if action.gripper >= 0.5 else 0.0
            if gripper is not None and cmd_state != gripper_command_state:
                width = (cfg.gripper.open_width_m if cmd_state == 1.0
                         else cfg.gripper.close_width_m)
                logger.info(f"  gripper → {'OPEN' if cmd_state == 1.0 else 'CLOSE'}")
                gripper.move_width(width, wait=False)
                gripper_command_state = cmd_state

            rate.finish_tick()

    except KeyboardInterrupt:
        logger.info("Interrupted — stopping.")
    finally:
        tracker.stop()
        try:
            cam_hand.close()
        except Exception:
            pass
        try:
            cam_tp.close()
        except Exception:
            pass
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    main()
