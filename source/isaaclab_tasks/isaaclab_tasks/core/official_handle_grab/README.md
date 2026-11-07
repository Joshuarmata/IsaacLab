# Isaac-HandleGrab / Isaac-HandleGrab-Camera

The handle-grab scene (SO-101, tray, bag lying on its side, strap handle, D415) trained the way NVIDIA's SO-101 vial
task is trained: small relative joint steps, dense shaping with latched milestones, resets from a physics-validated
dataset, an RSL-RL PPO **state teacher**, then a DAgger-style distilled **overhead-camera student**.

The scene comes from `core/official_handle_grab/scene/handle_grab_scene.py`, so both tasks and
`Isaac-HandleGrab-Direct` share one scene definition. The direct task is unchanged.

## Pipeline

```bash
# 0. Calibrate by hand first: see "Calibrate before you train" below.
# 1. Physics-validated reset dataset (GPU). Writes reset/handle_grab_reset_dataset.pt; training fails without it.
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator

# 1b. Watch the rows it wrote, one attempt at a time (home, pregrasp, grasp, grasp_closed); the vial's view-so101-resets.
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit

# 2. Smoke test and throughput.
uv run python scripts/environments/zero_agent.py --task=Isaac-HandleGrab --num_envs=16 --max_steps=300
uv run python scripts/benchmarks/runtime.py --task=Isaac-HandleGrab --num_envs=1024

# 3. State teacher (PPO).
uv run python scripts/reinforcement_learning/train.py --task=Isaac-HandleGrab

# 4. Camera student, distilled from the teacher's actor checkpoint.
uv run python scripts/reinforcement_learning/train.py --task=Isaac-HandleGrab-Camera \
    --agent=rsl_rl_distillation_cfg_entry_point --checkpoint=<path/to/teacher/model_XXXX.pt>

# Optional: the camera policy trained from scratch instead (slower, the vial task's own comparison).
uv run python scripts/reinforcement_learning/train.py --task=Isaac-HandleGrab-Camera

# Watch a policy (play mode: every episode starts at home, bags in dataset order).
uv run python scripts/reinforcement_learning/play.py --task=Isaac-HandleGrab --checkpoint=<path>
```

Presets combine: `presets=sim2real`, `presets=pinch_only`, `presets=sim2real,pinch_only`.

## Calibrate before you train

Do this once, by hand, before step 1 of the pipeline, and again whenever you move the arm, the bag, the tray or the
jaw. It takes about 30 minutes and needs the Isaac Sim window (`--viz kit`).

### Why this has to come first

PPO never sees the scene's geometry. It sees numbers: joint angles, the tip position, the distance to the strap, contact
flags. Every one of those numbers is computed from a handful of hand-entered constants, and a wrong constant does not
raise an error. It quietly teaches the wrong thing:

- **The reset dataset is the only place the task starts from.** Training samples rows from it and nothing else. Its rows
come from inverse kinematics (IK) that must put the tip on the strap. If the strap is out of the arm's reach, IK
stalls, every attempt is rejected and there is no dataset. If it is half reachable, the dataset holds only the easy
bag poses and the policy is blind to the rest. (This happened: the arm's 90 degree roll was missing, the arm lay on its
side, the strap was out of reach and every attempt failed.)
- **The approach reward measures distance from** `tip_pos_w` **to the recorded pinch point.** `tip_pos_w` is the wrist link
plus `EE_TIP_OFFSET`. If that offset or the recorded pinch is off by 2 cm, PPO learns to put the wrong spot on the
strap, collects the shaping reward and never produces a grasp.
- `holding` **uses the strap zone in the bag frame, the jaw's closed angle and contact points.** If the zone, the closed
angle or the jaw's insert axis is wrong, grasps that look right in the viewer never count, or grasps that are wrong
count and get latched into the milestones.
- **A policy trained against a wrong calibration cannot be fixed afterwards.** It has to be retrained, which costs hours
on the GPU. The checks below cost minutes.

### Vocabulary

- **Forward kinematics (FK)**: joint angles in, link poses out. The simulator does this every step.
- **Inverse kinematics (IK)**: a wanted wrist pose in, joint angles out. The generator copies the vial generator's
recipe (see "How the IK works" below): Levenberg-Marquardt on a position-first cost, 64 noisy starts around the
recorded joints, a pregrasp solve first and then the grasp.
- **Reachability**: IK can only find what the arm can physically reach inside its joint limits. If the error stops
falling above a few millimetres, the pose is out of reach, not badly seeded.

### Step 1. Look at the arm

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator \
  --viz kit --num_envs 1 --attempts 1 --max_batches 1 --output /tmp/handle_grab_check.pt
```

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator --viz kit --num_envs 1 --attempts 1 --max_batches 1 --output /tmp/check.pt
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit --dataset /tmp/check.pt
```

`--output` matters: a one-attempt dataset written to the default path would be loaded by training. The window closes
when the run ends; raise `--max_batches` to watch longer. Check by eye:

- The arm stands upright on the tray wall behind the bag and faces the tray. If it lies on its side or faces away,
stop: `ARM_BASE_ROT` in `scene/handle_grab_scene.py` is wrong for this asset.
- It moves home, then pregrasp, then grasp, then closes the jaw. The jaw tip should end up straddling the strap, with
the strap between the teeth and the bag body untouched.

Then play the dataset the run wrote, one row at a time (home, pregrasp, grasp, grasp_closed), the way the vial's
`view-so101-resets` does:

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator --viz kit --num_envs 1 --attempts 1 --max_batches 1 --output /tmp/check.pt
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit --dataset /tmp/check.pt
```

### Step 2. Check forward kinematics against the numbers

The first printed block for environment 0 is a free FK test. Placing the arm at the recorded joints puts the tip at a
fixed place whatever the bag does, so you can compare it with a value worked out independently:


| Printed line               | Right (arm upright)          | Wrong (arm on its side)       |
| -------------------------- | ---------------------------- | ----------------------------- |
| `tip at recorded joints`   | about (-0.001, 0.052, 0.026) | about (-0.001, -0.166, 0.312) |
| `home_tip_err_mm` (median) | about 380                    | about 670                     |
| `seed_tip_err_mm` (median) | below 50                     | about 370                     |


`strap target tip` is the recorded pinch on the bag as it lies now (for the nominal bag, about (0.002, 0.054, 0.037)).
`seed_tip_err_mm` is the distance from that to the tip at the recorded joints. The generator refuses to continue above
50 mm. Everything is in the environment frame, in metres.

If these are wrong, no amount of IK tuning helps: fix the mount (`ARM_BASE_ROT`, `ARM_BASE_POS`) first.

### Step 3. Check the IK

Read these in the batch printout (all values are medians over the batch, with the min and max beside them):

- `IK median position error [m] by iteration, unperturbed start`: the pregrasp solve's start that is the recorded joints. It should fall
below 0.005 within a few tens of iterations. Flat above 0.01 means this pose is out of reach from the recorded
posture; the noisy starts may still find it.
- `IK took ... s`: the cost of the multi-start search. If it is too slow, lower `--ik_seeds` (see below).
- `ik_valid_starts`: how many of the starts passed every test. Many valid starts means the pose is comfortable; one or
two means it only just works; zero means `ik_grasp_failed` / `ik_pregrasp_failed`.
- `ik_best_pos_err_mm`, `ik_pos_err_mm`, `ik_rot_err_rad`: the best any start did, and the chosen one's errors. A large
`ik_best_pos_err_mm` means out of reach; a small one with `ik_grasp_failed` / `ik_pregrasp_failed` means rotation or joint margin rejected it.
- `pregrasp_ik_best_pos_err_mm`, `pregrasp_ik_rot_err_rad`, `pregrasp_ik_valid_starts`, `pregrasp_ik_joint_margin_rad`: the
same readings for the pregrasp pose (6 cm back along the insert axis), which is solved first. The pregrasp and grasp
are counted separately, so a pregrasp-only failure is visible in `rejected`. The `ik_*` rows above are the grasp's.
- `ik_joint_delta_rad`: how far the chosen solution is from the recorded joints. Large values (above about 2 rad) mean a
different arm configuration (elbow flip, wrist flip), which the physics may not be able to reach from home.
- `ik_joint_margin_rad`: the closest any arm joint comes to its limit. It should be at least 0.03.
- `track_err_rad`: the gap between the IK joints and the joints the physics reached. Below about 0.05 rad the arm
follows its commands. Large means the arm was blocked (tray wall, bag) or is too slow: lower `JOINT_SPEED` in the
generator and look at the viewer.

Some `ik_grasp_failed` / `ik_pregrasp_failed` is normal for bag poses at the edge of the spawn box. If more than about a third fail, shrink the box
(`--y_half`, `--x_half`) or move `BAG_SPAWN_XY` closer to the arm. The handle must stay within about 0.30 m of the base.

#### How the IK works

This is the vial generator's recipe (`place_vial/reset/generator.py`), run on the live PhysX arm instead of a separate
Newton model:

1. **Target.** The wanted tip pose is the recorded pinch on the bag as it lies now. It is turned into a wrist pose by
  subtracting `EE_TIP_OFFSET`. The pregrasp (the vial's "overhead" pose, which it puts 3 cm above the vial) is the same
   pose backed off `--pregrasp_backoff` metres along the jaw's insert axis. With `--orientation recorded` (the default)
   the orientation is the recorded jaw orientation, which follows the bag's heading.
2. **Cost.** Position error counts 100 times as much as rotation error (`IK_POSITION_WEIGHT`, `IK_ROTATION_WEIGHT`),
  because the arm cannot reach every position with every orientation.
3. **Optimizer.** Levenberg-Marquardt, with a damping factor per environment that grows when a step does not help.
  Each iteration writes the candidate joints into PhysX and reads the wrist pose and Jacobian back.
4. **Starts.** `--ik_seeds` starts per pose: the recorded joints, plus that many minus one copies with Gaussian noise
  of `--ik_noise_std` radians on each arm joint.
5. **Validity.** A start counts only if position is within `--ik_position_tolerance` (the pregrasp uses 1.5 cm, like
  the vial's intermediate waypoints), rotation within `--ik_rotation_tolerance_deg`, and every arm joint at least
   `--ik_joint_margin` from its limit.
6. **Two solves, in the vial's order.** The pregrasp is solved first, from the recorded joints, and takes a random valid
  start (the vial's `randomize_solution`).
   The grasp is then solved for the orientation the pregrasp solution actually reached, from the pregrasp joints, and
   takes the valid start nearest them. That keeps both on one arm configuration and avoids wrist flips. A failed solve
   rejects the attempt as `ik_pregrasp_failed` or `ik_grasp_failed`.
7. **Physics.** The solution is only a command. The arm is driven home, pregrasp, grasp in PhysX and the grasp must pass
  the checks.

Two things the vial does that did not carry over. It derives the grasp orientation from the vial's geometry; the bag
has no such axis, so `--orientation vertical` tries the nearest equivalent (the recording with its insert axis made
exactly vertical). Offline that is reachable for only about 20 percent of bag poses at 18 degrees and 88 percent at 30
degrees, because the arm settles back toward the recorded 13 degree lean; use it only with
`--ik_rotation_tolerance_deg 30`. And random choice among valid starts gave no spread offline: every noisy start that
converges lands on the same arm configuration, so it varies the start state only as much as the bag pose does. The vial
gets its variety from the recorded waypoints of a real episode, which we do not have.

Defaults match the vial generator (64 starts, 120 iterations, noise 0.65, 5 mm, 18 degrees, margin 0.03). One deviation:
the search runs inside the joint limits pulled in by the margin, where the vial only rejects solutions that end near a
limit. The recorded grasp leaves `revolute_4` (wrist pitch) only 0.22 rad from its upper stop, so a pregrasp hovering
above it needs that joint at the stop, and every start was rejected. Searching inside the inset limits lets the
optimizer use the 18 degree rotation tolerance to stay clear. The cost is that the wrist can be tilted up to 18 degrees
from the recording, which the physics then checks.

#### Tuning the IK

Change one thing at a time and rerun the Step 1 command with `--num_envs 16`. Compare `ik_valid_starts`, `ik_grasp_failed` / `ik_pregrasp_failed`
and the final accepted count.


| Symptom                                                                     | Likely cause                           | Change                                                                                   |
| --------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ik_grasp_failed` / `ik_pregrasp_failed` high, `ik_best_pos_err_mm` large   | Pose out of reach                      | Shrink `--y_half` or move the bag nearer; shorten `--pregrasp_backoff`                   |
| `ik_grasp_failed` / `ik_pregrasp_failed` high, `ik_best_pos_err_mm` under 5 | Rotation or joint margin rejected it   | Raise `--ik_rotation_tolerance_deg` (try 25), or lower `--ik_joint_margin` (try 0.01)    |
| `ik_valid_starts` is 1 or 2                                                 | Only the unperturbed start works       | Raise `--ik_noise_std` (try 0.9), or more `--ik_iterations`                              |
| `ik_joint_delta_rad` above 2                                                | A flipped arm configuration was chosen | Lower `--ik_noise_std` (try 0.4) so starts stay near the recording                       |
| `IK took` is minutes                                                        | Too many starts times iterations       | `--ik_seeds 16` or `--ik_iterations 60`; the unperturbed start usually solves most poses |
| `track_err_rad` large but IK fine                                           | The arm cannot follow the command      | Lower `JOINT_SPEED`; check for a tray or bag collision in the viewer                     |
| Accepted but `grasp_tip_err_mm` over 10                                     | Physics did not reach the IK pose      | Same as above; also check `closed_tip_err_mm`                                            |


Loosening tolerances raises the accept rate but moves the grasp further from the recorded pinch. The physics checks
decide whether the loosened grasps are real, so read `rejected by reason` after every change.

### Step 4. Check the constants the rewards use

Look at the grasp phase in the viewer and compare with the printed stats:


| Constant (where)                                            | What it does                                           | How to check                                                                                                    |
| ----------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `EE_TIP_OFFSET`, `JAW_INSERT_AXIS` (`handle_grab_scene.py`) | Where `tip_pos_w` is; which way the pregrasp backs off | `grasp_tip_err_mm` and `closed_tip_err_mm` stay within about 10 mm; at the pregrasp the jaw points at the strap |
| `GRIPPER_OPEN_POS`, `GRIPPER_CLOSED_POS`                    | Open and closed jaw angles                             | `jaw_pos_rad` ends near -0.33 holding the strap (-0.38 when empty)                                              |
| Strap zone (`HANDLE_ZONE_*` in `mdp/task.py`)               | Which contact points count as the strap                | `rigid_jaw_on_strap` and `moving_jaw_on_strap` are both 1 for accepted attempts                                 |
| `ARM_HOME_JOINT_POS`, `BAG_REST_Z`                          | Home pose; where the bag sits                          | `bag_shift_mm`, `bag_lift_mm` and `bag_speed_mps` are about zero before the grasp                               |


Accepted attempts need `peak_lower_force_n` at most 3 and `holding_streak` of 10. If every rejection is `not_holding`
while the viewer shows a good grasp, the zone or closed-angle constants are the suspect.

### Step 5. Re-record the pinch only if steps 2 and 4 say so

The recorded pinch lives in `scene/grasp_goal.json`. Re-record it only if the tip lands off the strap even
though the arm is mounted right. The recorder is `scene/grasp_pose_tool.py`. It runs the
direct task, which uses this scene's `ARM_CFG`, so the arm stands at the same mount as the manager task.

#### What the vial approach can't give you

The viewer doesn't reload when you edit a file, and the vial's doesn't either. The vial workflow is: edit the numbers in
`generator.py`, rerun the generator, then view the result. Our equivalent is to edit `grasp_poses.json` and `--record`
the seed with `grasp_pose_tool.py` (the live-reload tool above), then rerun the generator.

### Step 6. Generate the real dataset and gate on it

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator --viz kit --num_envs 16 --attempts 16
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator   # then, headless, 256 attempts
```

Move on to the zero-agent smoke test only when the first run accepts most attempts and the second reports at least a few
hundred. Keep the `rejected by reason` line from each run: it is the record of what the physics validated.

## Task


| Name        | Meaning                                                                                                                                                              |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **holding** | Jaw within 0.10 rad of closed **and** the rigid jaw and the moving jaw each have their mean contact point on the bag inside the strap zone (bag frame:               |
| **grasped** | `holding` for 3 consecutive steps. Latches.                                                                                                                          |
| **lifted**  | Bag root at least 1 cm above the lowest height it has had this episode, while grasped and holding. Latches. The reference is the bag's own height, never the hand's. |
| **success** | `holding` and lifted for 5 s (150 steps) in a row. Ends the episode.                                                                                                 |
| **dropped** | After the grasp, not `holding` for 15 consecutive steps (0.5 s). A single lost step is not a drop.                                                                   |


`presets=pinch_only` is the fallback task: success is the closed jaw with the strap between the teeth for 5 s, with no
lift. It also drops the lift milestone and starts only from before-the-grasp phases.

Actions are relative joint steps: 0.033 rad on joints 1 to 6, 0.02 rad on the jaw. A zero action holds still.
The sim runs at 60 Hz, the policy at 30 Hz, episodes are 20 s.

## Rewards


| Term                            | Weight          | Why                                                                                                                                                                                                                                                                  |
| ------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `approach_progress`             | 1               | Decrease in tip-to-strap distance, per centimetre, until grasped. A difference of potentials: same gradient far away as close, no reward for hovering. This is what lets a camera policy find the approach from home.                                                |
| `held_goal`                     | 0.1             | Paid while grasped and holding, scaled by how far the bag has risen toward 1 cm. Dense signal through the 5 s hold.                                                                                                                                                  |
| `milestones`                    | 10              | Grasp x1, lift x2, once per episode. Milestones a `grasp_closed` reset starts with are not paid.                                                                                                                                                                     |
| `success`                       | 200             | Confirmed 5 s hold.                                                                                                                                                                                                                                                  |
| `bag_lost`                      | -50             | Dropping the bag or knocking it out of the tray (`is_terminated_term`).                                                                                                                                                                                              |
| `lower_body_contact`            | 0 → -4          | Force on the bag body below the strap beyond 1 N, saturating 5 N later. Jaws on the strap and top rim are free. The weight is ramped by a curriculum over 16000 env steps (about 250 PPO iterations): at full weight from step 0 the policy learns to avoid the bag. |
| `action_rate`, `joint_velocity` | -0.002, -0.0002 | Smoothness.                                                                                                                                                                                                                                                          |
| `joint_limits`                  | -1              | Joints 1 to 6 only; the jaw is meant to sit on its stop.                                                                                                                                                                                                             |


Terminations: `success`, `bag_dropped`, `bag_out_of_bounds` (outside the tray), `unstable_robot`, `time_out`.
The vial task's rack-specific terms are not used.

## Observations

- **Teacher actor** (`policy`): joint position, velocity, target, previous action, end-effector pose, bag pose, strap
point in the base frame, distance/lift/bag-speed/jaw-closed features, milestone flags.
- **Critic**: the same plus jaw contact flags and lower-body force (privileged).
- **Camera student**: two stacked overhead 160 x 90 RGB frames (uniform noise) and noisy proprioception. The teacher
group (`teacher_state`) is identical to `policy`, so the teacher checkpoint's input layout matches.

## Reset dataset

Four phases, one row each per accepted attempt: **home** (arm at home, jaw open, bag at rest), **pregrasp** (tip 6 cm
back along the approach axis, jaw open), **grasp** (tip on the recorded pinch, jaw open), **grasp_closed** (jaws shut on
the strap, bag still resting). Lifting is not a phase: the first lift has to be discovered.

Training samples the four phases equally. Evaluation and play start from **home** only, walking the home rows in order.

The generator places the bag, solves IK for the recorded pinch (`scene/grasp_goal.json`), drives the arm there
**in PhysX through the task's own action interface**, and keeps an attempt only if the physics ends in a `holding` grasp
for the last 10 steps with peak lower-body force of at most 3 N, the bag moved at most 3 cm and raised at most 5 mm, and
the bag nearly still. Each rejection reason is counted and printed. Rows store measured joints and the bag pose in the
environment frame.

**Spawn box.** Defaults are +-3 cm in x, +-4 cm in y and +-20 degrees of yaw around the scene's nominal spot
(`--x_half --y_half --yaw_half_deg`). The nominal spot is only 14 mm from the tray's near wall, so poses whose rotated
footprint would come within 5 mm of a wall are resampled. Effectively y covers about -3.9 cm to +1 cm around the
nominal, not the full +-4 cm. Widen it toward +y only if the arm can still reach (the handle must stay within about
0.30 m of the base).

## Evaluation

`evaluation.evaluate(env, policy, episodes_per_env=2)` runs complete episodes from home (build the env from
`resolve_task_config(..., play_mode=True)`) and returns grasp, lift, success, drop and out-of-bounds rates, mean time
to success and lower-body force. Training logs `Metrics/grasp_rate`, `Metrics/lift_rate`, `Metrics/success_rate` and
`Safety/peak_lower_body_force_n`, but those average over all four reset phases and overstate the home task.

## Where this differs from the vial task

- **Arm base rotation.** `so101_physics.usda` stands this Y-up arm up with a 90 degree roll about X on `/Arm/Mount`.
`ARM_BASE_ROT` is the 180 degree yaw on `/Arm` above that mount, which turns the upright arm toward the bag. The
vial's robot is already Z-up, so its scene only needs +90 degrees of yaw. `grasp_goal.json` was recorded with this
mount. The direct task uses the same `ARM_CFG`.
- The goal is a held strap, not a placement, so there is no placement flag; `grasped` and `lifted` take that role.
- Resets come from IK-generated rows validated in PhysX, not from a teleoperated demonstration.
- The success hold time is 5 s, not a single placement check.
- PPO runs 1024 environments (vial: 4096), so the state teacher defaults to 1500 iterations instead of 800 to see a
comparable number of samples. This is a starting point, not a tuned value.
- The camera is a regular `Camera` (the D415 asset), not a tiled camera, so it is likely the slowest part of the camera
task. Not measured.

## Part A. The NVIDIA vial example, file by file and step by step

This part describes only NVIDIA's SO-101 vial example (`isaac-sim/IsaacLabTutorial`, package `isaaclab_tutorial`).
Nothing in it is about the handle-grab project; Part B starts from it.

### A0. What the example is

An SO-101 arm picks up a vial lying on a desk and places it upright in any of the four holes of a rack. The pieces, in
the order they are used:

1. A **task** (`IsaacTutorial-Place-Vial-SO101`): a scene, an action interface, observations, rewards, terminations.
2. A **reset dataset** (`reset_poses.pt`): 1024 physics-validated start states, 128 for each of eight phases of the
  task. Every training episode starts from one of them.
3. A **teacher**: an RSL-RL PPO policy that sees the full state.
4. A **student** (`IsaacTutorial-Place-Vial-SO101-Camera`): a wrist-camera policy distilled from the teacher. This is the
  policy that runs on the real robot.
5. **Evaluation** and **sim-to-real** scripts.

```text
workshop constants ──► reset generator (IK + physics) ──► reset_poses.pt
(demonstrated seed,                                          │
 recorded waypoints)                                         ▼
env_cfg.py + mdp/*.py ──────────────────────────────► PPO teacher (4096 envs, 800 iterations)
                                                             │ checkpoint
                                                             ▼
camera_env_cfg.py ───────────────────────────────────► distilled camera student (1024 envs)
                                                             │ export
                                                             ▼
                                              LEAPP bundle ──► deploy.py on the real robot
```

**IK is only used in the reset generator.** Neither the teacher nor the student calls IK. They learn from start
states that IK helped build.

### A1. Every file

Paths are relative to the repository root.

**Repository root**


| File                                                                 | What it is                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                                                          | Setup, task table, train/evaluate commands, task design, layout.                                                                                                                                                                                                                                                             |
| `pyproject.toml`                                                     | Package metadata and dependencies (`isaaclab==3.0.0`, `torch`). Registers the `isaaclab.tasks` entry point (`isaaclab_tutorial.tasks`) so Isaac Lab discovers the tasks, and two console scripts: `generate-so101-resets` and `view-so101-resets`. Optional extras: `video`, `wandb`, `leapp`, and the PhysX/OVRTX backends. |
| `uv.lock`                                                            | Locked versions for `uv sync --locked`.                                                                                                                                                                                                                                                                                      |
| `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`, `LICENSE` | Git LFS rules (the USD and dataset files are LFS pointers until `git lfs pull`), ignore rules, lint hooks, license.                                                                                                                                                                                                          |
| `media/demo.gif`, `media/demo.mp4`                                   | The demo video shown in the README.                                                                                                                                                                                                                                                                                          |


`config/` **and** `docs/`


| File                                  | What it is                                                                                                                                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config/so101_start_pose.json`        | The six joint angles the real robot must be homed to before running a policy, with a tolerance of 0.035 rad.                                                                                         |
| `config/so101_joint_map.example.json` | A template of per-joint scale, offset and limits that maps LeRobot's degrees and gripper percent to simulation radians. It ships with `"verified": false`; you copy it and verify it for your robot. |
| `docs/SIM2REAL.md`                    | The robot-transfer guide: training with `presets=sim2real`, physical setup, calibration, joint map, homing, export, running, safety.                                                                 |


`src/isaaclab_tutorial/assets/`


| File                                                | What it is                                                                                                                                                                                      |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assets/__init__.py`                                | Defines the paths `VIAL_USD`, `VIAL_LABELED_USD`, `RACK_USD`, `DESK_USD`, `RESET_DATASET`, and `validate_assets()`, which lists missing files.                                                  |
| `assets/workshop/*.usda`, `assets/workshop/LICENSE` | The desk, rack (`rack.usda`, `rack_simple.usda`), and vial variants (`vial.usda`, `vial_labeled.usda` for the camera task, `vial_opaque.usda`), and their license. The USD files are LFS files. |
| `assets/reset_poses.pt`                             | The shipped reset dataset. LFS file.                                                                                                                                                            |


`src/isaaclab_tutorial/tasks/place_vial/config/so101/` **(the task definitions)**


| File                                | What it is                                                                                                                                                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                       | Registers the two gym ids and tells each where its env config and agent configs live.                                                                                                                                           |
| `env_cfg.py`                        | The state task: scene, spawn functions, physics presets, actions, observations, events, rewards, terminations, the `sim2real` preset. Also holds the workshop constants (`WORKSHOP_INITIAL_JOINT_POSITION`, gripper positions). |
| `camera_env_cfg.py`                 | The camera task. It inherits the state task and adds the wrist camera, the image and proprioception observation groups, and appearance randomization.                                                                           |
| `agents/rsl_rl_ppo_cfg.py`          | PPO settings for the state teacher and for a camera policy trained from scratch.                                                                                                                                                |
| `agents/rsl_rl_distillation_cfg.py` | Distillation settings for the camera student, plus `BoundedTeacherDistillation`.                                                                                                                                                |


`src/isaaclab_tutorial/tasks/place_vial/mdp/` **(the MDP terms)**


| File                          | What it is                                                                                                                                                                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task.py`                     | Shared geometry constants (vial extent, grasp offset, rack heights, thresholds) and helper functions: jaw contacts, fingertip positions, grasp centre, vial in the rack frame, gripper openness.                                         |
| `geometry.py`                 | Pure tensor maths with no simulator: rack-frame transforms, hole offsets, vertical alignment, lowest point of a cylinder, a symmetric axial keypoint error, and the test that a lying vial overlaps the rack.                            |
| `progress.py`                 | `PlacementProgress`: the per-episode latches `grasped`, `lifted`, `inserted`, `success`, with counters, shared by rewards, terminations and observations.                                                                                |
| `observations.py`             | Observation terms: joint targets, encoder-biased joint readings, vial-in-rack position, placement features, progress flags, jaw contact flags, and `wrist_rgb` (the camera image with per-episode intrinsics and photometric variation). |
| `rewards.py`                  | `approach_progress`, `held_goal`, `milestones`, `success`, `jaw_support_force`.                                                                                                                                                          |
| `terminations.py`             | `placement_success` (updates the milestones every step and ends the episode on success) and `vial_dropped`.                                                                                                                              |
| `events.py`                   | `ResetFromDataset` (the reset), `randomize_joint_viscous_friction`, `randomize_encoder_bias`, `randomize_wrist_camera_mount`.                                                                                                            |
| `__init__.pyi`, `__init__.py` | Re-export the terms so configs can write `mdp.approach_progress`.                                                                                                                                                                        |


`src/isaaclab_tutorial/tasks/place_vial/reset/`


| File            | What it is                                                                                                                                                                                                        |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `curriculum.py` | The eight phases in order (home, pregrasp, grasp, lift, reorient, transport, insert, release) and the two sampling mixes: `ALL_PHASES` (uniform, for training) and `CANONICAL_START` (home only, for evaluation). |
| `dataset.py`    | The dataset format: schema version 2, `save_reset_dataset` (validates, writes atomically, stores a content hash) and `load_reset_dataset` (re-validates on load).                                                 |
| `generator.py`  | Builds the dataset. About 2,200 lines: constants, `GeneratorCfg`, the `_Generator` class, and the `generate_main` and `view_main` entry points.                                                                   |


`src/isaaclab_tutorial/utils/`


| File               | What it is                                                                                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation.py`    | `install_episode_counter`: makes `play` stop after exactly one episode per environment and print an `SO101_EVAL_RESULT` JSON line. |
| `calibrate.py`     | Runs LeRobot's interactive motor calibration for the real follower arm.                                                            |
| `inspect_so101.py` | Reads the follower's joint coordinates without touching torque, to check the joint map.                                            |
| `home_so101.py`    | Slow, supervised homing of the real arm to `so101_start_pose.json`.                                                                |
| `export_leapp.py`  | Packages the trained camera actor as a LEAPP bundle and checks it against the original on CPU.                                     |
| `deploy.py`        | Runs the bundle on the real robot: camera frames in, joint targets out, with safety checks.                                        |


`tests/`: 15 files, each guarding one area. `test_config` (task registration and structure), `test_assets`,
`test_geometry`, `test_progress`, `test_rewards`, `test_any_hole` (success in every hole), `test_reset_dataset`,
`test_reset_generator` (config validation, phase and lift rules, waypoints cover every loaded phase),
`test_camera_observations`, `test_camera_projection`, `test_evaluation`, `test_leapp_export`, `test_deploy`,
`test_home_so101`, `test_startup` (importing the config loads no USD).

### A2. Step 0: set up and test

```bash
git lfs pull
uv sync --locked
uv run pytest -q
```

`git lfs pull` replaces the LFS pointer files (the USD assets and `reset_poses.pt`) with the real files.
`uv sync --locked` installs exactly the versions in `uv.lock`. The tests need no GPU.

### A3. Step 1: how the tasks are found

`pyproject.toml` declares `[project.entry-points."isaaclab.tasks"] isaaclab_tutorial = "isaaclab_tutorial.tasks"`. When
Isaac Lab starts, it imports that package. `config/so101/__init__.py` then calls `gym.register` twice. Each call names
an `env_cfg_entry_point` (the config class to build the environment from) and the agent config entry points, so the
command line only has to say `--task IsaacTutorial-Place-Vial-SO101`.

### A4. Step 2: the scene, physics and presets (`env_cfg.py`)

- **Scene (**`SO101SceneCfg`**).** The robot, a vial lying horizontally on the desk, a rack, the desk, two lights, and
contact sensors: one on the fixed jaw, one on the moving jaw (filtered to the vial), one on the vial (filtered to the
rack). Robot base position is (-0.05, 0, 0), rotated +90 degrees about world Z.
- **Spawn functions.** `_spawn_so101` loads the robot USD, binds a contact material, applies the workshop's elbow
correction of 6.4 degrees, sets the gripper's calibrated range, and removes every collider except the two jaws.
`_spawn_usd_with_contact_material` does the contact-material binding for the vial, rack and desk.
- **Physics.** `PhysicsCfg` is a preset: Newton MJWarp by default (`presets=newton_mjwarp`), or PhysX. Physics runs at
120 Hz (`sim.dt = 1/120`) and the policy at 30 Hz (`decimation = 4`). Episodes last 20 s, 4096 environments by default,
0.9 m apart.
- **Presets.** `preset(default=..., sim2real=...)` values change with the command line. `presets=sim2real` enables
randomization for robot transfer (A13).

### A5. Step 3: the MDP (actions, observations, events, rewards, terminations)

**Actions (**`ActionsCfg`**).** Relative joint-position targets: at every physics step the target is the *measured* joint
position plus `action x scale`. The scale is 0.033 rad for the five arm joints and 0.02 rad for the gripper. A zero
action holds still. This matches the real servo interface.

**Observations (**`ObservationsCfg`**).**

- `policy` (the teacher's actor): joint positions, velocities and targets, the previous action, the gripper's pose and
the vial's pose in the robot base frame, the vial in the rack frame, placement features (distance to the nearest
hole, lowest vial point, uprightness, vial speed), and the three latched milestone flags.
- `critic`: the same plus the privileged jaw-contact flags. The critic is only used in training.
- The camera task adds `wrist_rgb`, `proprioception` and `teacher_state` groups (A9).

**Events (**`EventsCfg`**).** At startup, vial friction and mass are randomized. On every reset,
`ResetFromDataset` writes a row of the reset dataset into the scene (A7).

**Rewards (**`RewardsCfg`**).**


| Term                                            | Weight               | What it pays                                                                                                                                                |
| ----------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `approach_progress`                             | 1                    | The decrease in distance from the pad centre to the vial's cap, until grasped. A difference of potentials, so it gives the same gradient far away as close. |
| `held_goal`                                     | 0.1                  | While held and not yet inserted, closeness of the vial to the insertion pose above the nearest hole.                                                        |
| `milestones`                                    | 10                   | Grasp, lift and insertion, once each per episode, in the ratio 1:2:4. A milestone present on the first step after a reset is not paid.                      |
| `success`                                       | 200                  | A confirmed placement.                                                                                                                                      |
| `vial_lost`                                     | -50                  | Dropping the vial or knocking it out of bounds.                                                                                                             |
| `action_rate`, `joint_velocity`, `joint_limits` | -0.002, -0.0002, -1  | Smoothness and staying off the limits.                                                                                                                      |
| `jaw_support_force`                             | -8 (`sim2real` only) | A jaw pushing on the desk or mat.                                                                                                                           |


**Terminations (**`TerminationsCfg`**).** `placement_success`, `vial_dropped`, `vial_out_of_bounds`, `unstable_robot`
(both from the lift task), `time_out`.

**How the terms share state.** `progress.py` keeps one `PlacementProgress` on the environment. The `placement_success`
termination updates it every step; the rewards and the `progress_flags` observation only read it. That is why the
once-per-episode milestone rewards are Markov: the latch is in the observation.

### A6. Step 4: what success means

Success is a vial seated upright in a hole (within a box around the hole centre), released (both jaws off it and the
gripper open), and almost still, for ten consecutive control steps. Grasp needs both jaws on the vial off the desk for
two steps; lift is the vial's lowest point clearing the rack rim; insertion is the upright vial's tip below the rim
within 12 mm of a hole centre.

### A7. Step 5: the reset dataset

`reset_poses.pt` is a dictionary with a format name, a schema version, the phase names, `states`, generator and
validation metadata, and a SHA-256 of the contents. `states` holds, per row: `joint_position` (6), `joint_target` (6),
`vial_pose` (7), `phase`, `difficulty` (0 to 1), `grasped`, `lifted`.

There are eight phases, 128 rows each: home (called `approach` in the dataset), pregrasp, grasp, lift, reorient,
transport, insert, release. `ResetFromDataset` (in `events.py`) turns the phase weights into per-row weights, draws
rows, writes the joint positions and targets into the robot, the pose into the vial and rack, and records
whether the row already counts as grasped or lifted, so the milestone rewards are not paid for a state the episode
started in. Training uses `ALL_PHASES` (every phase equally). Play and evaluation use `CANONICAL_START` (home only, rows
in order).

### A8. Step 6: the reset generator (`reset/generator.py`)

This is the only file that uses IK. Everything below happens once, offline, when someone regenerates the dataset.

```bash
uv run generate-so101-resets --output outputs/reset_poses.pt --visualizer none presets=newton_mjwarp
uv run view-so101-resets --dataset outputs/reset_poses.pt --visualizer newton presets=newton_mjwarp
```

**A8.1 The recorded numbers (top of the file).** Two constants came from a real robot at the workshop:

- `WORKSHOP_PREGRASP_JOINT_POSITION`: six joint angles of a real pregrasp. It is the **starting guess** for the first IK
solve, not a target.
- `WORKSHOP_TASK_WAYPOINTS`: `{phase: (pose, pose, pose)}`, three recorded joint poses each, from one successful real
episode. Phases 3 (lift) and 4 (reorient) replay them. The entries for phases 5, 6 and 7 are present, but those phases are
generated another way (A8.5). Both constants get the same +6.4 degree elbow correction as the robot asset.

**A8.2** `GeneratorCfg`**.** 128 poses per phase; batch size 1024 environments; 32,768 attempts per phase before giving up;
IK settings: 64 starts, 120 iterations, noise 0.65 rad, position tolerance 5 mm, rotation tolerance 18 degrees, joint
margin 0.03 rad; `transition_steps` 120 (steps for one joint-space move) and `contact_settle_steps` 90.

**A8.3 Building the IK (**`_build_ik`**).** The generator builds a separate Newton model of the arm from the simulation's
robot, finds the `gripper` body and the six joint coordinates, and creates an `isaaclab_newton.controllers.ik.NewtonIKSolver`:
Levenberg-Marquardt, analytic Jacobian, Gaussian seed sampler, 64 seeds, with a pose objective on the gripper body
(offset to the grasp centre) and a joint-limit objective.

**A8.4 The approach and the pregrasp (**`_approach_candidate`**,** `_solve_pregrasp`**).**

1. Start at the workshop home pose; settle; place the vial at a random pose (about +-3 cm and +-4 cm, a modest heading).
2. **Overhead pose, IK solve 1.** Target position is the vial's grasp point 3 cm higher. Target orientation is
  **derived from the vial's geometry**: the pads run along the vial axis, the jaws close along a horizontal direction
   perpendicular to it, and the approach axis follows. The solver starts from `WORKSHOP_PREGRASP_JOINT_POSITION` plus 63
   noisy copies. A start is **valid** when the position error is within tolerance, the closing axis is nearly
   perpendicular to the vial axis, the pad axis is aligned with the vial axis, and every arm joint is at least the margin
   from its limit. One valid start is chosen **at random** (`randomize_solution=True`).
3. **Pregrasp pose, IK solve 2.** Target position is the grasp point itself. The orientation is the one solve 1 actually
  achieved. It starts from solve 1's joints. Among valid starts it picks the lowest score, which combines alignment,
   distance from the starting joints (continuity) and position error.
4. The arm is placed at the overhead pose, the vial is parked away, and the arm **sweeps home to overhead to pregrasp** in
  PhysX; then the vial is put back, the jaw closes and the grasp settles. Rows for phases 0 and 1 are snapshots along
   this sweep.

**A8.5 Later phases (**`_connected_candidate`**).** Each phase continues from validated states of the one before it.


| Phase              | How the arm moves                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 2 grasp            | Close the jaw on the vial (`grasp_close_steps`), then prove the grasp by lifting it 1 cm.                                  |
| 3 lift, 4 reorient | Replay `WORKSHOP_TASK_WAYPOINTS` (A8.6), then close the residual tilt with Cartesian feedback.                             |
| 5 transport        | IK again: raise the vial, then translate it over the rack through Cartesian waypoints, keeping the vial-to-gripper offset. |
| 6 insert           | IK again: `_solve_vial_pose` finds the gripper pose that puts the vial at the target pose, then interpolate to it.         |
| 7 release          | Branch from verified held insertions and open the jaw to the release position.                                             |


**A8.6 Interpolation.** The generator never teleports the arm between waypoints during a rollout. It uses `torch.lerp`:

- `_move_robot(target, steps)` (about line 1277) loops `step = 1..steps`, sets the command to `lerp(start, target, step / steps)` and simulates one step each time.
- `_follow_workshop_segment` (about line 1284) walks the three recorded poses of a phase, lerping between consecutive
ones, scaled by a `progress` value, and checks that every waypoint is inside the soft joint limits.
- `_approach_candidate` lerps home to overhead to pregrasp; `_connected_candidate` lerps the insertion and the release.

**A8.7 Physics validation (**`_valid`**).** After each rollout the generator checks the state the physics actually reached.
Examples: finite and inside joint limits, calm joints, both jaws touching, the vial held off the table, lifted by the
right amount, still, upright, centred between the pads, low drift. The IK validity from A8.4 is part of the check.

**A8.8 Seed banks and quotas.** Validated endpoints of the grasp, lift, reorient and transport phases are cached
(`_append_seed_bank`) and restored (`_restore_seed_bank`) as the starting point of the next phase, so the trajectory is
connected. `generate` fills each phase to 128 rows by rejection sampling and raises an error if it cannot. `difficulty`
records progress along the whole task from 0 to 1.

**A8.9 Output.** `save_reset_dataset` writes `reset_poses.pt`. `view_main` (`view-so101-resets`) writes each row back into
the scene and holds it for 45 steps so you can watch every state.

### A9. Step 7: the camera task (`camera_env_cfg.py`)

It inherits the state task and adds a wrist camera that renders 80 x 60 pixels, resampled to a 64 x 48 policy view so
varied intrinsics still see real pixels. Observation groups: `wrist_rgb` (two stacked frames), `proprioception`
(encoder-coordinate joint positions, velocities, targets, previous action, all with noise), `teacher_state` (identical to
the teacher's `policy` group, so the teacher's checkpoint fits), and `critic`. Under `presets=sim2real` it also
randomizes colours and the camera mount.

### A10. Step 8: smoke test

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp
```

Builds the environment and steps it with zero actions. If this works, the config, assets and dataset load.

### A11. Step 9: train the teacher, then distil the student

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint /abs/path/to/state/model_799.pt --visualizer none presets=newton_mjwarp,newton_renderer
```

- **Teacher** (`rsl_rl_ppo_cfg.py`): actor and critic are MLPs of 256, 256, 128 with ELU and observation normalization.
64 steps per environment per iteration, 800 iterations, a model saved every 50. PPO: clip 0.2, entropy 0.005, 5 epochs,
8 minibatches, learning rate 3e-4 (adaptive), gamma 0.995, lambda 0.95. The action standard deviation is clamped
between 0.05 and 0.3.
- **Student** (`rsl_rl_distillation_cfg.py`): a small CNN (16, 32, 32 channels) plus an MLP of 256, 128, trained by
DAgger-style distillation. The student acts and the frozen teacher labels every state the student visits; the loss is
MSE and the labels are clipped to [-1, 1], which `BoundedTeacherDistillation` does. The teacher block must mirror the
teacher's actor so its checkpoint loads. 32 steps per environment, up to 1600 iterations.
- The camera policy can also be trained from scratch with PPO (`--agent rsl_rl_ppo_cfg_entry_point`, 2500 iterations).

Checkpoints go to `logs/rsl_rl/`.

### A12. Step 10: evaluate

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 1024 --checkpoint /abs/path/to/model.pt --visualizer none --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter presets=newton_mjwarp
```

`play_mode()` in `env_cfg.py` makes every episode start from home, reads dataset rows in order, and (without
`sim2real`) pins the vial's friction and mass. `evaluation.install_episode_counter` patches the environment wrapper so
play stops after exactly one episode per environment (1024 by default, `SO101_EVALUATION_EPISODES` changes it) and
prints one `SO101_EVAL_RESULT` JSON line; `SO101_EVALUATION_OUTPUT=path.json` saves per-episode outcomes. Do not pass
`--deterministic` on this pin.

### A13. Step 11: sim-to-real and the real robot

`presets=sim2real` randomizes friction, vial mass and diameter, joint friction, armature and damping, drive gains,
encoder bias (+-0.01 rad), vial and rack placement and mat height, seats the base on the real 35 mm tabletop, uses
larger shoulder and elbow action scales, and starts every episode from home. The camera task adds colours, camera mount,
focal length, principal point, distortion, exposure and blur. The recommended order is: teacher on the nominal task,
continue the teacher under `sim2real`, distil the student under `sim2real`.

On the robot (`docs/SIM2REAL.md`, sections 3 to 7), each script is run with `uv run --script` and pins its own
dependencies:

1. `calibrate.py`: LeRobot motor calibration.
2. Fill in `config/so101_joint_map.example.json` and check it with `inspect_so101.py --watch`.
3. `home_so101.py`: move the real arm to `so101_start_pose.json`.
4. `export_leapp.py`: turn the trained actor's TorchScript export into a LEAPP bundle.
5. `deploy.py`: run the bundle; it reports actions without moving unless given `--execute` and a verified joint map.

### A14. Quick reference: seed, waypoints, IK and interpolation


| Question                                  | Answer                                                                                                                                                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The demonstrated pregrasp seed            | `WORKSHOP_PREGRASP_JOINT_POSITION` in `reset/generator.py` (near line 109). Six hard-coded numbers recorded on a real robot. There is no tool in the repository that generates them; you change the numbers in the file and rerun the generator. |
| The sparse commands from one real episode | `WORKSHOP_TASK_WAYPOINTS` in the same file (near line 119). Replayed by `_follow_workshop_segment`.                                                                                                                                              |
| The IK with 64 noisy seeds                | `NewtonIKSolver` (`isaaclab_newton/controllers/ik/newton_ik_solver.py` in Isaac Lab), wrapping `newton.ik`. Configured in `_build_ik` and used in `_solve_pregrasp` and `_solve_vial_pose` in `reset/generator.py`.                              |
| The interpolation                         | `torch.lerp` in `_move_robot`, `_follow_workshop_segment`, `_approach_candidate` and `_connected_candidate`, all in `reset/generator.py`.                                                                                                        |


## Part B. The handle-grab project, step by step, related to the vial example

This part lays out our project in the same order as Part A. "A8.4" means "Part A, section A8.4".

### B0. The one-paragraph version

We kept the vial's pipeline: a task with relative joint actions and latched milestones, a physics-validated reset
dataset, a PPO state teacher, and a distilled camera student. Three things had to change because we do not have the
vial's inputs. We have no pre-calibrated asset (the arm mount had to be calibrated, B3), no real-robot episode to replay
(our start states stop at the closed grasp, so the lift is discovered by RL), and no Newton model (our IK runs on the
live PhysX arm).

### B1. Every file

Paths are under `source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/` unless they start with `core/` (relative to `source/isaaclab_tasks/isaaclab_tasks/`) or `source/`.


| Our file                                                            | Vial counterpart           | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                                                       | `config/so101/__init__.py` | Registers `Isaac-HandleGrab` and `Isaac-HandleGrab-Camera`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `env_cfg.py`                                                        | `env_cfg.py`               | The state task: scene, actions, observations, events, rewards, terminations, curriculum, `sim2real` and `pinch_only` presets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `camera_env_cfg.py`                                                 | `camera_env_cfg.py`        | The overhead-camera task (D415, RGB only).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `agents/rsl_rl_ppo_cfg.py`                                          | same                       | PPO for the teacher (1500 iterations) and a from-scratch camera policy (2500).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `agents/rsl_rl_distillation_cfg.py`                                 | same                       | Distillation for the camera student (1600 iterations), with `BoundedTeacherDistillation`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `mdp/task.py`                                                       | `mdp/task.py`              | Strap zone, contacts, `holding`, the tip and strap positions, `ARM_BASE_ROT`, and the loader for `grasp_goal.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `mdp/progress.py`                                                   | `mdp/progress.py`          | `GrabProgress`: `grasped`, `lifted`, `success`, `dropped`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `mdp/rewards.py`, `terminations.py`, `observations.py`, `events.py` | same names                 | The terms. We add `lower_body_contact` and `bag_dropped`, and drop the rack terms.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `mdp/curriculums.py`                                                | none                       | `linear_weight_ramp`, which grows the lower-body penalty from 0 to full weight over 16000 steps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `reset/curriculum.py`                                               | `reset/curriculum.py`      | Four phases: home, pregrasp, grasp, grasp_closed. - It defines the four labels home, pregrasp, grasp and grasp_closed, in that order (their ids in the dataset).- It also defines the phase mixes that say how often each phase is sampled. For example, training uses all four equally, and evaluation uses only home.- It contains no joint angles or arm poses.- The actual poses come from two places:- The recorded grasp is in grasp_goal.json and grasp_poses.json. That gives the jaw's position and orientation at the strap.- The home joint angles come from the robot config.                         |
| `reset/dataset.py`                                                  | `reset/dataset.py`         | The dataset format (simpler: no content hash). Where the generator's output is stored and read back. There are three jobs here:- Storage- Phase ID - Attempt Number- The 7 arm joint angles - The bag pose- save() and load() write and read the .pt file. load() rejects a file whose versions or phases don't match. - Checking - validate_phase_weights raises an error if a requested phase has no rows - Sampling - Sample_rows picks a phase by the weights from the file curriculum.py, then picks a random row from that phase. The generator file below calls ResetDataset.save() at the very end. |
| `reset/generator.py`                                                | `reset/generator.py`       | Builds the dataset. About 800 lines, against the vial's 2,200. Mostly inverse-kinematics, but it does more. - Solves IK for the pregrasp pose, then for the grasp pose, then closes the jaw. Those are the arm states for the later phases.- It spawns the bag at random positions in the spawn box.- It drives the arm through the sequence in the live physics sim.- It keeps a state only if the physics checks pass (for example, the bag didn't move and the jaws gripped it).- It writes everything to a file.- It is slow and runs once, offline                                                            |
| `reset/viewer.py`                                                   | `view_main`                | Plays the dataset rows in the viewer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `evaluation.py`                                                     | `utils/evaluation.py`      | `evaluate(env, policy, episodes_per_env)`. Not a `play` callback.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `README.md`                                                         | `README.md`                | This file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `scene/__init__.py`, `scene/handle_grab_scene.py`                   | `env_cfg.py` scene section | `HandleGrabSceneCfg`: sandbox, SO-101, bag, contact sensor, and lights. The camera is added by `camera_env_cfg.py`. The direct task builds its own scene from the same asset configs.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `scene/assets/__init__.py`, `scene/assets/`                         | `assets/__init__.py`       | File paths for the wrappers the scene loads: bag, tray, arm, strap ribbon, and `build_handle_assets.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `scene/grasp_poses.json`, `scene/grasp_goal.json`                   | `WORKSHOP_*` constants     | The recorded arm waypoints and the pinch the generator aims at (B4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `scene/grasp_pose_tool.py`                                          | none                       | Records those files. Live mode reloads `scene/grasp_poses.json` when you save it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `source/isaaclab_tasks/test/core/test_official_handle_grab.py`      | `tests/`                   | All our tests, in one file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |


We have no counterpart for the vial's `utils/calibrate.py`, `inspect_so101.py`, `home_so101.py`, `export_leapp.py`,
`deploy.py`, or `docs/SIM2REAL.md`. The real-robot half of the vial example is not built for this project.

### Josh Edit B2. Step 0: Setting the scene

The vial puts the spawned world on one class, `SO101SceneCfg`, and the file paths in `scene/assets/__init__.py` 

Ours is the same split. Paths are in `scene/assets/__init__.py`. 

Insiide /scene/assets/**init**.py we define: 

```python
ASSET_ROOT = Path(__file__).parent
SANDBOX_USD = ASSET_ROOT / "sandbox_tray.usda"
ARM_USD = ASSET_ROOT / "so101_physics.usda"
BAG_USD = ASSET_ROOT / "bag_single_body.usda"
BAG_BODY_USD = ASSET_ROOT / "bag_body_only.usda"
HANDLE_USD = ASSET_ROOT / "bag_handle_ribbon.usda"
HANDLE_INFO = ASSET_ROOT / "bag_handle_info.json"
JAW_POINTS = ASSET_ROOT / "jaw_points.json"
```

The spawned objects are the fields of `HandleGrabSceneCfg` in `scene/handle_grab_scene.py`. `env_cfg.py` uses that class as `scene`. It's here where we build our full scene. 

We can use the following command to view the scene we just built: 

```bash
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit \
```

Later we're going to add noise to the joints, x,y and yaw jitter for bag, and so on. An example of a command that takes these into account (if you want to just look at the environment) would be: 

```bash
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit \
  env.use_demo_residual=false \
  env.arm_joint_noise=0 \
  env.bag_x_range=[0,0] \
  env.bag_y_range=[0,0] \
  env.bag_yaw_range=[0,0]
```

Note: The direct task / student trainer builds its own scene from the same asset configs and adds the D415. The manager scene leaves the camera off, the way the vial adds its camera in `camera_env_cfg`.

### Josh Edit B3: Name the task and check numbers 

In the last step we only defined the world. Now we need to give what we just built a name that Isaac Lab can look up. Once we do this we need to check that thename we just gave points at the world we just built. 

gym.register is the entry that looks up all the information we built. 

handle_grab_[scene.py](http://scene.py) does not run a robot. It stores numbers and file paths: the tray USD, the arm USD, the bag USD, the base position (0, -0.20, 0.06), and the 180 degree yaw. env_[cfg.py](http://cfg.py) wraps that scene in a larger description called HandleGrabEnvCfg: how big a joint step is, what counts as success, what the reward pays. That object is still just data in memory.

Three programs consume it, at three different times:

Kit is the Isaac Sim application. It is a 3D simulator. When you pass --viz kit, a program starts Kit, asks for the task by name, and Kit creates the prims, turns on PhysX, and draws the window. The zero-agent command is one of these programs. It asked for Isaac-HandleGrab-Direct, which is a different registration, the older direct-control task. The manager task you are building is a second name, Isaac-HandleGrab, pointing at HandleGrabEnvCfg.

The generator is reset/[generator.py](http://generator.py). It also starts Kit, but headless or with a window, and it uses the simulation for a job: move the bag, solve for joint angles that put the jaw on the strap, step physics, and save the joint angles that actually worked. It has to load the same scene the viewer showed, or it will solve against a different arm position than the one you looked at.

Training is scripts/reinforcement_learning/[train.py](http://train.py). It starts many copies of the same world, runs a policy, and updates the policy from rewards. It loads the world by the same task name. The policy never sees handle_grab_[scene.py](http://scene.py). It sees whatever config that name resolves to.

**Why the tests come next, and how they run**

The lookup can succeed and still hand Kit the wrong world. A typo in ARM_BASE_POS, or a scene class that is no longer the one env_[cfg.py](http://cfg.py) uses, still produces a config. Kit would faithfully spawn the wrong one. The tests build the config the same way Kit will, then read the fields, and they do it without starting Kit.

From the Isaac Lab repo root:

uv run python -m pytest source/isaaclab_tasks/test/core/test_official_handle_[grab.py](http://grab.py) source/isaaclab_tasks/test/core/test_env_cfg_no_forbidden_[imports.py](http://imports.py) -q

Pytest is a test runner. It imports the two files, calls every function whose name starts with test_, and prints a pass or a failure. Importing them imports isaaclab_tasks, which runs gym.register. No --viz flag is passed, so Kit stays shut and there is no window.

The first file connects the phone book to the scene. One test calls resolve_task_config("Isaac-HandleGrab"). That function does the lookup a training command will do: find the id, import HandleGrabEnvCfg, return it. The test then reads cfg.scene.robot.init_state and compares it with the constants in handle_grab_[scene.py](http://scene.py).

The position is (0, -0.20, 0.06), the base on the south wall.

The rotation is (0, 0, 1, 0), the 180 degree yaw written onto /Arm.

Combining that yaw with the 90 degree roll stored on /Arm/Mount in so101_physics.usda sends the arm's up axis to world +Z. That is the upright arm you saw.

The sandbox and bag paths are the USD files named in scene/assets/__init__.py.

If the lookup were broken, this test would fail before any of those comparisons, because there would be no config to read. If the lookup worked but pointed at a different scene, the numbers would not match. Either failure means the generator and training would build a different world from the one you inspected.

The same file also checks a rule that has no picture. GrabProgress is the object that will later decide success. The test creates it on the CPU and feeds it two fake environments for a dozen steps. In both, the bag height goes from 0 to 2 cm. Only the first environment is marked as holding the strap. The test expects grasp, lift, and success for the first, and all three flags off for the second. The scene can show a bag in the air. It cannot show whether the task would call that a success. This test is that decision, run on lists of numbers. The reward and the termination will call this same object during training, so a wrong decision here would train the policy to knock the bag upward.

The second file connects the scene to the way Kit starts. Isaac Lab builds HandleGrabEnvCfg before it opens Kit. It reads the config to decide which physics backend to launch. Building the config imports the scene module. The test starts a separate Python process, loads every task whose name begins with Isaac-, and writes down every library that import pulls in. pxr, omni, carb, isaacsim, and scipy have to stay out of that list. Those libraries belong to a running simulator. SciPy in particular loads OpenBLAS, which registers handlers that crash Kit during the fork() Kit uses to start. A scene file that imported one of them at the top would make the viewer die before drawing the tray. The test has to run while Kit is closed. With Kit already open, those imports are ordinary and the crash would not show up.

What you have when pytest passes

The name Isaac-HandleGrab resolves to HandleGrabEnvCfg. That config's scene is the tray, the upright arm, and the bag you built. The success rule does not treat a rising bag as a grasp. The config can be imported before Kit starts, which is the order Kit, the generator, and training all use. The simulator has not moved the jaw, and no pinch has been recorded. That recording is the next step, and it will ask for a task by name in the same way.





### Josh Edit B4: Record one pinch  



### Josh Edit B5: Play the hand waypoints once. 



### Josh Edit B6: Write the task 



### Josh Edit B7: Generate rows and watch 



### Josh Edit B8: Smoke-test environment 



### Josh Edit B9: Train state teacher  



### Josh Edit B10: Add Camera and Train the Student 



### Josh Edit B11: Evaluate   





