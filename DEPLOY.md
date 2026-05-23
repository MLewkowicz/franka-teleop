# Running 3D Diffuser Actor on the real Franka

Branch: `deploy/diffuser-actor` of `franka-teleop`. Read this end-to-end the
first time you deploy — there are decisions baked into the converter and
trainer that determine what the model expects to see at inference, and a few
things about the collection rig that the model is *not* robust to.

---

## 1. What the script actually does

`deploy_diffuser_actor.py` runs an MPC loop:

1. Grab one synchronized `(rgb, depth)` from each ZED via `ZedCamera.grab_frame()`.
2. Read the current end-effector pose from `CartesianImpedanceTracker.current_pose`.
3. Replicate the **exact** geometry from `LangSteer/scripts/convert_realworld_for_diffuser_actor.py`: center-crop 720×1280 → 720×720, resize to 200×200, unproject depth using the per-camera intrinsics (adjusted for the crop+resize), transform the per-pixel XYZ into the robot base frame — `T_cam2base` for the overhead camera, `T_cam2gripper ∘ T_gripper2base(t)` for the wrist camera.
4. Build a `core.types.Observation` with `rgb={"front": tp, "wrist": hand}`, `depth={"front": tp_xyz, "wrist": hand_xyz}` (the policy reuses `obs.depth` for per-pixel XYZ — this is the convention in `diffuser_actor_base.py`), and `ee_pose=[xyz(3), euler_XYZ(3), gripper_cmd(1)]`.
5. Call `policy.forward(obs)` → `Action` with `trajectory (20, 7)` absolute ee-poses (xyz + euler_XYZ + gripper bit) and a scalar `gripper` for step 0.
6. Clip `trajectory[0, :3]` to a base-frame workspace box (`deploy.workspace_lo/hi`).
7. Convert the first pose's euler → rotation matrix and call `tracker.set_target(Affine(pos, R))`.
8. If the predicted gripper bit flipped relative to the last commanded state, issue `gripper.move_width(...)`.
9. Loop at `deploy.control_hz` (default 10 Hz).

There is no trajectory queue — every tick re-plans from scratch. That's the point of MPC here: a fresh observation feeds a fresh 20-step prediction, you execute one step, throw the rest away, and re-plan. If inference becomes the bottleneck (forward >100 ms), drop `control_hz` until you have headroom.

---

## 2. Prereqs

### Hardware
- Franka Emika Panda with FCI enabled, reachable at `cfg.robot.ip` (default `172.16.0.2`).
- Two ZED 2i cameras:
    - **wrist** (`hand`), ZED 2i serial `10986074`, rigidly mounted on the gripper. *Must be in the same position as during data collection — the extrinsics depend on this.*
    - **third-person**, serial `37820861`, statically mounted overhead.
- Robotiq gripper proxy running on `cfg.gripper.host:cfg.gripper.port` (defaults `172.16.0.1:18812`).
- 3Dconnexion SpaceMouse (for operator control). If absent the script logs a warning and you'll have to Ctrl-C to stop — the script won't enable closed-loop control without it.
- A workstation with a recent NVIDIA GPU and CUDA. Inference on a 5090 is ~50-70 ms per tick; you can afford 10 Hz comfortably. On weaker GPUs drop to 5 Hz.

### Software stack on the robot machine
1. **`franka-teleop`** with this branch (`deploy/diffuser-actor`) checked out.
2. **`net_franky` server** running and reachable at `cfg.net_franky.ip:port` (defaults `172.16.0.1:18812`). Same server `teleop.py` uses.
3. **`LangSteer`** cloned somewhere accessible. The default install is:
   ```bash
   cd LangSteer
   uv sync                # installs the full main deps + the `dev` group
   ```
   That gets you everything the policy needs: torch, transformers (for CLIP), diffusers, einops, hydra-core, numpy, scipy, h5py, opencv-python-headless, plus accelerate / huggingface-hub / blosc / dill etc. as transitive deps.

   *uv groups vs extras*: the only optional groups are `dev`, `calvin`, `diffusion`, `sagemaker` — pass `--group <name>` to opt into one. There are no `[project.optional-dependencies]` (a.k.a. extras), so `--extra <anything>` will always error. `transformers` is a regular dependency and is already installed by bare `uv sync`.

   *If `uv sync` fails on `calvin-env`*: that's a git dep with PyBullet + NumPy-2.0 compatibility issues that occasionally won't build. The inference path does not import `calvin_env`, so the fix is to comment out the `"calvin-env @ git+..."` line inside `dependencies = [...]` in `pyproject.toml` and rerun `uv sync`. (Don't remove `calvin-env` from the `[dependency-groups].calvin` block — that one's behind `--group calvin` and is already skipped by default.)

   *Running the deploy script*: always invoke through the venv, either `uv run python deploy_diffuser_actor.py ...` or `source LangSteer/.venv/bin/activate` first. Plain `python deploy_diffuser_actor.py` will use system Python and fail with `No module named 'torch'`.

   *Which LangSteer branch*: the `realworld/data-support` branch is only required if you also re-train on this machine. For pure deployment, `main` or `refactoring` is enough — the inference code paths are identical.
4. **ZED SDK** installed and visible to the venv (the `pyzed` Python bindings). Both cameras must be claimed by this process — no `ZED Explorer` or other ZED-using process running concurrently.
5. **`threed_mouse` Python package** (the SpaceMouse client `teleop.py` uses).

### Files you must have on disk
| File | Where | Why |
|---|---|---|
| trained checkpoint `last.pth` | anywhere, point to it via `deploy.checkpoint` | the weights |
| `data/extrinsics_hand.json` | `cfg.deploy.extrinsics_hand` | wrist intrinsics + `T_cam2gripper` |
| `data/extrinsics_third_person.json` | `cfg.deploy.extrinsics_third_person` | overhead intrinsics + `T_cam2base` |
| `conf/deploy_policy.yaml` | already in this branch | model architecture mirror; **must match the checkpoint** |
| `conf/config.yaml` | already in this branch | robot/cameras/gripper config |

**Important:** the extrinsics files must be the same ones in scope when the training data was recorded. If you re-calibrate (especially the wrist camera) between data collection and deployment, the model's PCD inputs will be misaligned and predictions will be silently wrong. If you must re-calibrate, retrain. There's no halfway position.

---

## 3. Pre-run checklist

Walk this every time, especially on the first run after any rig change.

- [ ] Robot powered, FCI enabled, no errors on the desk panel.
- [ ] `net_franky` server up (`echo > /dev/tcp/172.16.0.1/18812` returns ok).
- [ ] Robotiq gripper proxy up (check `cfg.gripper.host:cfg.gripper.port`).
- [ ] Both ZEDs detected: `lsusb | grep -i ZED` shows two devices; no other process holding them.
- [ ] Cameras physically in the **same** position as during collection — same wrist mount, same overhead pose. If anything moved, re-calibrate AND retrain.
- [ ] Wine glass placed in the workspace, roughly where it was during demos. The model has only seen a small workspace region — do not place the glass somewhere it never appeared at training time.
- [ ] Workspace box (`deploy.workspace_lo` / `deploy.workspace_hi`) is sane for what's on the table. Default `[0.30, -0.35, 0.02] → [0.85, 0.20, 0.85]` is sized from the recorded teleop's ee range plus a buffer — tighten it before letting the robot loose.
- [ ] Operator at the E-stop. The workspace clip is **not** a safety system; it's a sanity guard that prevents the policy from commanding pathological poses. It does not protect against gripper-side collisions, controller errors, or the policy decking the table at full stiffness.
- [ ] `LANGSTEER_PATH` set or `deploy.langsteer_path` overridden on the CLI.
- [ ] Checkpoint path resolved and readable.

---

## 4. Launching

Bare minimum:
```bash
export LANGSTEER_PATH=/path/to/LangSteer
cd franka-teleop
uv run python deploy_diffuser_actor.py \
    deploy.checkpoint=/path/to/last.pth
```

Common overrides:
```bash
uv run python deploy_diffuser_actor.py \
    deploy.checkpoint=/path/to/last.pth \
    deploy.langsteer_path=$HOME/repos/LangSteer \
    deploy.control_hz=8 \
    deploy.workspace_lo='[0.40,-0.25,0.05]' \
    deploy.workspace_hi='[0.75, 0.10, 0.70]' \
    teleop.translational_stiffness=200 \
    teleop.rotational_stiffness=10
```

The script first resets to `cfg.teleop.reset_joint_config` synchronously, then opens the Cartesian impedance tracker and waits for you to enable.

---

## 5. Operator controls

SpaceMouse only — there is no keyboard control loop.

| Button | Action |
|---|---|
| **LEFT short tap** | toggle ENABLED. While disabled, the script keeps grabbing frames and reading robot state (so you can preview live latencies) but does not call `tracker.set_target`. |
| **RIGHT short tap** | advance to the next stage. Stages cycle `grasp glass → place glass → exit`. At each boundary the script calls `policy.set_primitive(..)` + `policy.reset()` to flush the gripper history buffer so the model doesn't see stale poses across stages. |

Long-press semantics from `teleop.py` (reset to start config) are **not** wired into the deploy script. If you need to bail, hit the E-stop, then Ctrl-C the script, then re-launch.

---

## 6. What you should see in the logs

The first 5 forward passes are logged in full:
```
forward[1] 64.3ms  traj[0]=[ 0.534 -0.041  0.211 -1.871  1.099 -1.157  1.000]  gripper=1.00
```
- `forward[N] xx.x ms` — measure this against your tick period. At 10 Hz you have 100 ms.
- `traj[0]` — first predicted absolute ee pose (xyz + euler_XYZ + gripper bit). Pos values should be in the robot's workspace (~0.3-0.8 in x), euler within ~[-π, π], gripper close to 0 or 1.
- After step 5 the per-tick spam quiets down; rely on the SpaceMouse + the robot's actual motion to monitor health.

Stage transitions print:
```
[stage 1] place glass
  gripper → CLOSE
```

Critical errors will surface as either `tracker.set_target failed` (disables control automatically; you'll need to clear errors and re-enable) or `Camera grab failed — skipping tick` (transient; investigate if it repeats).

---

## 7. Caveats baked into the training data (read this part)

Three findings from auditing `franka-teleop`'s collection code that matter at deployment time:

1. **The recorded `gripper_open` is a command, not measured state.** It changes the instant the user pressed the button, *before* the physical gripper has finished closing. We trained the model to predict that command, so this is consistent — but it means the model's idea of "now I'm grasping" is ~50-200 ms ahead of physical contact. Be patient with the RIGHT-button stage advance: tap it once you've *visually* confirmed the gripper is closed around the glass, not the moment you see "gripper → CLOSE" in the log.

2. **Training data is from `replay.py`, not the original demos.** `replay.py` re-executes the original SpaceMouse demos under `JointImpedanceTracker`, recording the resulting ee poses. The original demos used `CartesianImpedanceTracker` (the SpaceMouse drives Cartesian targets). So the recorded ee trajectories reflect joint-impedance tracking dynamics, not the original Cartesian-impedance dynamics. We deploy with `CartesianImpedanceTracker` (closer match to the *intent* of the demo and the natural pairing with ee-pose targets). Expect minor dynamics differences; if the controller tracks predicted poses sluggishly, the first lever is raising `teleop.translational_stiffness` / `teleop.rotational_stiffness`.

3. **Camera frame rate is ~5-10 Hz despite `fps=30` in the config.** Not configurable — it's just what the ZED SDK + threading sustains on this rig. Our deployment uses **synchronous** `grab_frame()` so we always get *some* frame, but it's bounded by ZED throughput. If `control_hz` is set higher than the cameras can sustain, `grab_frame()` will block and slow the loop down to the ZED's natural rate. 10 Hz is safe; 30 Hz is not.

---

## 8. Safety

The workspace box clip in `deploy_diffuser_actor.py` clips `trajectory[0, :3]` element-wise to `[workspace_lo, workspace_hi]` before the Cartesian set_target. This catches gross policy failures (model commanding `z = -10`), not subtle ones. It does not:
- Prevent the wrist from colliding with the table.
- Detect that the gripper has snagged something it shouldn't have.
- Catch oscillations or fast jumps within the workspace.
- Replace the physical E-stop.

Always have a hand on the E-stop on the first runs. Tighten `workspace_*` before any unattended operation. The Cartesian stiffness defaults in `cfg.teleop` (`translational_stiffness=100`, `rotational_stiffness=5`) are deliberately compliant — you'll feel the robot push back if it commands something a human is blocking, which is the right default for early testing.

---

## 9. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named 'policies.diffuser_actor'` | `LANGSTEER_PATH` wrong / not set | export `LANGSTEER_PATH` or pass `deploy.langsteer_path=...` |
| `RuntimeError: deploy.langsteer_path=... does not look like a LangSteer checkout` | path points to wrong dir | should be the repo root (contains `policies/diffuser_actor.py`) |
| `Checkpoint has N MISSING keys` warning at load | `deploy_policy.yaml` doesn't match the trained architecture | check `num_primitives` / `num_objects` / `embedding_dim` against the trainer config |
| `tracker.set_target failed: ControlException` | robot in error state (E-stop, joint limit, force violation) | clear errors on the desk panel; the script auto-disables, so toggle LEFT to re-enable after recovery |
| `Camera grab failed — skipping tick` (repeated) | ZED dropped off USB, or another process owns it | check `lsusb`; restart any stale Python process holding the ZED |
| Model commands wildly wrong poses on first tick | gripper history buffer not flushed across episodes | `policy.reset()` happens automatically at stage transitions; if you re-launch without restarting the script, the buffer carries state — restart the script |
| Tracking lags badly | inference latency > tick period | drop `deploy.control_hz`; first 5 forwards are logged with timings |
| Gripper never closes despite `gripper → CLOSE` in log | Robotiq proxy not reachable | check `cfg.gripper.host:port`; `move_width(..., wait=False)` swallows connection errors silently |
| Model produces a credible trajectory but the wrist PCD looks misaligned in any debug view | extrinsics file changed between collection and now | re-collect + retrain, or restore the original extrinsics JSON |

---

## 10. Files map

```
franka-teleop/                            (branch: deploy/diffuser-actor)
├── deploy_diffuser_actor.py              ← MPC loop
├── conf/
│   ├── config.yaml                       ← + deploy: section
│   └── deploy_policy.yaml                ← policy architecture (must match checkpoint)
├── clear_franka/
│   └── diffuser_actor_io.py              ← per-tick geometry: crop+resize, depth→camera→base
├── data/                                  (.gitignored)
│   ├── extrinsics_hand.json              ← K, T_cam2gripper
│   └── extrinsics_third_person.json      ← K, T_cam2base
└── DEPLOY.md                              ← this file

LangSteer/                                 (branch: main or realworld/data-support)
└── policies/
    ├── diffuser_actor.py                 ← factory
    ├── diffuser_actor_base.py            ← Observation → obs tensors, gripper history, action denormalization
    └── diffuser_actor_primitive_object.py← variant the deploy script uses
```

Inference depends on `LangSteer/policies/*` and `LangSteer/training/policies/diffuser_actor/preprocessing/pytorch3d_transforms.py`. Nothing else from LangSteer is on the inference path.

---

## 11. What's not yet implemented (deliberate scope limits)

- **Automatic stage transition.** Currently the operator triggers grasp→place on the SpaceMouse RIGHT button. A force-sensor watch on the Robotiq feedback or a fixed-delay-after-close-command would replace this, but for first deployments manual triggering is the safest.
- **Multi-object support.** The vocab is `{glass: 0}` only; the embedding tables are sized to 1 object. To deploy on a different object you must (a) record demos of it, (b) rerun `convert_realworld_for_diffuser_actor.py --auto_segment_object <name>`, (c) add it to `object_vocab` in the training config and `deploy_policy.yaml` with `num_objects` bumped, (d) retrain.
- **VoxPoser/steering at deployment.** LangSteer's `steering/voxposer_steering.py` is designed for exactly this stage-transition problem but is currently untested with real cameras. Stick to the bare MPC loop until the basic glass run is robust.
- **Chunk-based execution.** We execute only step 0 of each predicted trajectory. If the inference cost ever becomes prohibitive (e.g. moving to a smaller GPU), a chunk-of-K alternative would amortize forward calls — the trajectory output already supports it; the loop would need a small rewrite.

---

## 12. Quick reference — what each magic number means

- `200` in `IMG_SIZE` (geometry helpers): the size CalvinDataset stored at. The model crops `[:, 20:180, 20:180]` to 160×160 at training and inference; never feed it 256×256 or 224×224 directly.
- `nhist=3` in `deploy_policy.yaml`: gripper history length. Three timesteps of `[pos(3), quat_wxyz(4)]` are stacked.
- `pred_horizon=20` and `interpolation_length=20`: keep these in sync with the trainer; the model predicts 20 future poses regardless of MPC step count.
- `gripper_loc_bounds=[[-0.30,-0.26,-0.22],[0.36,0.34,0.18]]`: normalization range for the *relative* trajectory deltas (not absolute workspace). Same values used at training; do not change at inference without retraining.
- `embedding_dim=192`, `num_vis_ins_attn_layers=2`, `fps_subsampling_factor=3`: model architecture. Must match the checkpoint.
- `cameras=["front","wrist"]`: camera order convention. `front=third_person`, `wrist=hand`. The `.dat` files were stored in this order; the policy assumes it.
- `rotation_parametrization="6D"`, `quaternion_format="wxyz"`: how the model represents rotation internally + the quaternion convention. Must match training.
