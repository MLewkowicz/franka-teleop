# Deploying a 3D Diffuser Actor checkpoint on the robot

Branch: `deploy/diffuser-actor`. Pull this branch on the robot machine.

## Layout added by this branch

| Path | What it is |
|---|---|
| `deploy_diffuser_actor.py` | MPC inference loop (Hydra-driven, mirrors `teleop.py` structure). |
| `clear_franka/diffuser_actor_io.py` | Live-frame → 200×200 RGB+PCD-in-base-frame helpers. Same geometry as the LangSteer converter. |
| `conf/deploy_policy.yaml` | Policy architecture yaml — sizes match the trained checkpoint (5-prim/8-obj CALVIN vocab → 2-prim/1-obj real-world). |
| `conf/config.yaml` | Adds a `deploy:` section (checkpoint path, extrinsics paths, control rate, workspace box). |

## Prereqs on the robot machine

1. LangSteer cloned somewhere accessible (`export LANGSTEER_PATH=/path/to/LangSteer` or set `deploy.langsteer_path`). For inference you only need its policy/model code + `torch`, `transformers`, `diffusers`, `einops`, `scipy`, `numpy`, `hydra-core`. The heavy optional groups (`calvin`, `diffusion`, `sagemaker`) can be skipped.
2. The trained checkpoint (`outputs/checkpoints/diffuser_actor_realworld_primitive_object/last.pth`) copied over.
3. The two extrinsics JSONs (`data/extrinsics_hand.json`, `data/extrinsics_third_person.json`) — these are *robot-rig-specific*, so use the same files the training data was collected with.
4. Standard franka-teleop stack: `net_franky` server reachable at `cfg.net_franky.ip:port`, ZEDs plugged in, Robotiq gripper proxy running.

## Launch

```bash
uv run python deploy_diffuser_actor.py \
    deploy.checkpoint=/path/to/last.pth \
    deploy.langsteer_path=$LANGSTEER_PATH
```

Override any `deploy.*` field on the CLI in the usual Hydra way.

## Operator controls (SpaceMouse)

- **LEFT short tap** — toggle ENABLED. While disabled the script keeps grabbing frames and reading robot state but does not send Cartesian targets; while enabled it streams predictions at `deploy.control_hz`.
- **RIGHT short tap** — advance to the next stage. Stages: `grasp glass` → `place glass` → exit. The script calls `policy.set_primitive(..)` + `policy.reset()` at the boundary to flush the gripper history buffer.

## Control loop

Every tick (`deploy.control_hz = 10` Hz default):

1. Grab one `(rgb, depth)` from each ZED via `ZedCamera.grab_frame()`.
2. Read current ee pose from `CartesianImpedanceTracker.current_pose`.
3. Build `T_gripper2base`, run the same crop+resize+depth-unprojection as the LangSteer converter, transform PCDs into the base frame (`T_cam2gripper` ∘ `T_gripper2base` for the wrist, `T_cam2base` for the overhead).
4. Construct a `core.types.Observation` with `rgb={'front': tp, 'wrist': hand}`, `depth={'front': tp_xyz, 'wrist': hand_xyz}`, `ee_pose = [xyz, euler_XYZ, gripper_cmd]`.
5. `action = policy.forward(obs)` → 20-step absolute trajectory `(20, 7)` + scalar gripper command.
6. Clip `action.trajectory[0, :3]` to the `deploy.workspace_*` box.
7. Convert euler → rotation matrix, send `tracker.set_target(Affine(pos, R))` for the first step only.
8. If `action.gripper` flipped relative to the last command, call `gripper.move_width(...)`.
9. Loop.

## Things to watch on the first runs

- **First-pose latency**: the script logs `forward[N] xx.x ms ...` for the first 5 inference calls. If that's above the tick period (100 ms at 10 Hz), drop `deploy.control_hz` until you have headroom.
- **Stage transition timing**: with two-button operation it's manual — wait until the gripper has visibly grasped the glass before the RIGHT tap to switch to `place`. A force-sensor trigger or a fixed delay after the close command would automate this; ask for it once you've validated the manual flow.
- **Domain gap from the joint-impedance replays**: the model learned ee-poses produced by `replay.py` under `JointImpedanceTracker`. Deploying through `CartesianImpedanceTracker` (chosen for closer match to the original demonstration controller) may produce subtly different dynamics — if you see the model commanding poses that look right but the controller tracks them sluggishly, raising the translational/rotational stiffness in `cfg.teleop.*` is the first lever.
- **Workspace clip**: `[0.30, -0.35, 0.02]` → `[0.85, 0.20, 0.85]` is generous; tighten before letting it run unattended.
