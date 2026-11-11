# Isaac-HandleGrab / Isaac-HandleGrab-Camera

> **Lift is shelved: the task is grasp-only for now.** Success is the strap held in the closed jaws for
> `HOLD_SECONDS` (5 s); the bag does not have to rise. The reset dataset has four phases (home, pregrasp, grasp,
> grasp_closing); `lift`, the lift milestone, `Metrics/lift_rate`, and the `pinch_only` preset are gone, because no
> grasp of the deformable strap could lift the bag. Why, what was tried, and the removed code are in
> [`LIFT_GRAVEYARD.md`](LIFT_GRAVEYARD.md). Lift, `require_lift`, and `pinch_only` mentions further down this README
> are historical. The vial's closed-grasp phase came back as `grasp_closing`, a jaw nearly closed on the strap; see
> "Known issue: the teacher never closed the jaw".

The handle-grab scene (SO-101, tray, bag lying on its side, strap handle, D415) trained the way NVIDIA's SO-101 vial
task is trained: small relative joint steps, dense shaping with latched milestones, resets from a physics-validated
dataset, an RSL-RL PPO **state teacher**, then a DAgger-style distilled **overhead-camera student**.

The scene comes from `core/official_handle_grab/scene/handle_grab_scene.py`, so both tasks and
`Isaac-HandleGrab-Direct` share one scene definition. The direct task is unchanged.

## What we changed / fixed and why

A running log, newest first. Each entry says what changed, why, and where to read more. Add new entries at the top.

| Date       | What we changed / fixed                                                                                                                                                                                                                                                                                         | Why                                                                                                                                                                                                                                           | Details                                                          |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 2026-11-10 | `deploy` compiles the image kernels before the timed loop, and the ONNX session uses one thread with spinning off. | On a CPU-only machine the default ONNX Runtime pool spin-waited on every core, the crop then took about 80 ms, and the 30 Hz loop stopped with "loop too slow". | "Sim to real" |
| 2026-11-10 | The trained student export is in the repo at `logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/` (`policy.onnx` and `policy.onnx.data`). `handle_grab/agents` skips registering its camera network when `rl_games` is not installed. The driver install includes `onnxruntime`. | The robot computer has no Isaac Sim training stack. Importing the tasks pulled in `rl_games`, and the exported weights were gitignored, so `deploy` could not start. | "Sim to real", `LIVE_RUN.md` |
| 2026-11-10 | `real/`: the student on the physical arm without ROS. `calibrate` (joint map from a hand-posed home), `twin` (the sim arm mirrors the real one, real and simulated camera overlaid), `home`, and `deploy` (dry run unless `--execute`). `bridge.py` builds the policy inputs with the simulator's own image code. `check_bridge` compares them, and ONNX against `policy.pt`, with the env's. | The policy is a 30 Hz loop of camera frame + joint angles in, joint targets out; ROS adds nothing for one arm and one camera. Every sim/real mismatch (joint signs, zero offsets, crop, normalization, action scaling) is checked in code before the arm moves. | "Sim to real"                                                    |
| 2026-11-10 | Camera task `presets=sim2real`: per-episode image augmentation (exposure, white balance, contrast, gamma, blur, noise) and D415-like depth error (scale, offset, noise growing with distance, dropout, holes at edges); camera pose jitter (2 cm, 3 degrees); light intensity, tint and direction; bag, strap, floor and wall colours re-drawn every 20 to 40 s. | The student had seen one look. The real camera, room light and objects will not match the render, so the student has to see enough variety that the real scene is just another sample.                                                       | "Sim to real"                                                    |
| 2026-11-10 | Camera student sees depth as well as RGB: an `overhead_depth` group (two stacked frames, clipped at 1 m and scaled to [0, 1], about 4 mm of noise, 0 for no return as on the real D415) with its own CNN. The image is 256 x 144 (was 160 x 90), and each CNN has a fourth stride-2 layer.                       | Depth gives the jaw's and strap's height directly, which one RGB view can only guess. At 160 x 90 the strap was two or three pixels wide.                                                                                                     | "The overhead camera (student)"                                  |
| 2026-11-10 | Camera moved: 0.70 m up, nearly straight down, tilted about 12 degrees in from +x. Camera task lights: dome 700 (was 800), key light 1800 (was 3000) with a 25 degree angle (was 2) from 35 degrees off vertical. The state and direct tasks keep the old camera and lights.                                     | Straight above the bag, the wrist hid the strap at the grasp; the tilt sees past it. Higher, the whole tray is in view. The old lights clipped the white floor. The new ones give soft shadows and no floor hotspot; RGB peaks at about 238. | "The overhead camera (student)"                                  |
| 2026-11-10 | `plot_training.py --experiment handle_grab_camera_distillation` plots or live-follows the student run; the noise panel also shows `Loss/behavior` (the student-teacher action loss).                                                                                                                            | Distillation has no PPO reward signal of its own; the action loss shows whether the student is catching up with the teacher.                                                                                                                | "Pipeline", step 4                                               |
| 2026-11-10 | `plot_training.py --live`: a window with the training graphs that redraws every 30 s (`--refresh`) while `train.py` runs, following the newest run.                                                                                                                                                            | To watch success, grasp rate and the reward terms climb during a run, and stop or tweak a run early, without waiting for it to finish.                                                                                                    | "Pipeline", step 3c                                              |
| 2026-11-10 | Replaced the `grasp_closed` reset phase (jaw already holding) with `grasp_closing`: jaw slid 15 mm onto the strap and closed to -0.10 rad, 0.2 rad short of holding. Every row is checked to hold once the jaw finishes closing. Dataset version 4.                                                                | Closed starts handed out free successes: the policy only had to not let go for 5 s. Training read 96% success while teaching "stay still" more than "grasp". Now the policy has to finish the close itself; held still, 0 of 32 starts hold. | "Follow-up: closed-grasp starts handed out free successes"       |
| 2026-11-10 | `reset_fade` curriculum: the `grasp_closing` share of starts falls from a quarter to none over the first 300 iterations.                                                                                                                                                                                         | The nearly closed starts teach the last part of the close; after that, every grasp should start further out (home, pregrasp, open-jaw grasp).                                                                                                | "Reset dataset"                                                  |
| 2026-11-10 | `success_by_start` logging: `Curriculum/success_by_start/<phase>`, a running success rate per start type.                                                                                                                                                                                                        | `Episode_Termination/success` mixes every start type and overstated how well the policy grasps from home. `home` is the number play and evaluation measure.                                                                                | "Evaluation"                                                     |
| 2026-11-10 | `plot_training.py`: training curves against PPO iteration (how episodes end, success by start type, grasp rate, every reward and penalty term, curriculum, noise, episode length), plus a printed summary of how quickly success climbed.                                                                          | To see how fast training is progressing and what to tweak, without reading TensorBoard by hand. Runs can be overlaid with `--compare`.                                                                                                    | "Pipeline", step 3b                                              |
| 2026-11-10 | Raised the grasp = true reward: `held_goal` 0.1 to 0.2, `milestones` 10 to 20 (up to 1.67 per episode, from 0.83).                                                                                                                                                                                              | The grasp should out-pay every shaping and placement term (approach was about 0.8). Only `success`, which is the same grasp kept for 5 s, is larger.                                                                                     | "Rewards"                                                        |
| 2026-11-10 | Three fixes for a teacher that never closed the jaw: a closed-grasp reset phase (since replaced, see above), the `jaw_closing` reward (weight 10, pays once per episode for closing at the strap), and the jaw step raised from 0.02 to 0.04 rad.                                                                    | 0 grasps in 850 iterations. The 0.9 rad close was far beyond the exploration noise (about 0.1 rad per episode), and no reward paid for closing before `holding` was already true.                                                       | "Known issue: the teacher never closed the jaw"                  |
| 2026-11-10 | Rewards for where the strap is held: `grasp_centered` (weight 0.1, top of the handle's arch over its sides) and `strap_depth` (weight 0.05, strap close to a point at the back of the mouth).                                                                                                                      | A grasp on top of the handle balances the bag; a strap deep in the jaws is held more securely than one at the tips.                                                                                                                         | "Rewards"                                                        |
| 2026-11-10 | The grasp boolean (`holding()` in `mdp/task.py`) now requires the jaw closed and the strap to cross the closed mouth from one side of the gripper to the other (`strap_in_mouth()`). Removed the contact-force and jaw-distance grasp detection.                                                                      | A strap only touching the box, poking out of its front, or lying along the jaws is not a grasp. The force and distance tests counted those.                                                                                                | "The grasp box (closed mouth)"                                   |
| 2026-11-10 | Grasp box (`closed_mouth` in `jaw_points.json`): four boxes filling the closed jaws from 25 to 65 mm in front of the hinge, carried by the rigid jaw. `scene/gap_viewer.py` draws it.                                                                                                                              | The earlier box sat too far toward the jaw tips and too small for the closed gripper, and counted the strap with the jaw open.                                                                                                             | "The grasp box (closed mouth)"                                   |

## Pipeline

```bash
# 0. Calibrate by hand first: see "Calibrate before you train" below.
# 1. Physics-validated reset dataset (GPU). Writes reset/handle_grab_reset_dataset.pt; training fails without it.
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator

# 1b. Watch the rows it wrote, one attempt at a time (home, pregrasp, grasp, grasp_closing); the vial's view-so101-resets.
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit

# 2. Smoke test and throughput.
uv run python scripts/environments/zero_agent.py --task=Isaac-HandleGrab --num_envs=16 --max_steps=300
uv run python scripts/benchmarks/runtime.py --task=Isaac-HandleGrab --num_envs=1024

# 3. State teacher (PPO).
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task=Isaac-HandleGrab --viz none

# 3b. Training curves (outcomes, grasp rate, every reward term, noise) against PPO iteration, from the TensorBoard log.
#     Writes <run>/training_curves.png and prints how quickly per-episode success climbed. Newest run by default;
#     --run <name>, --compare <name> ... to overlay runs, --per_episode for per-episode rewards, --smooth N.
uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training

# 3c. Live graphs while training, in a second terminal. A window redraws every --refresh seconds (default 30) and
#     follows the newest run, so it can be started before train.py. Close the window to stop.
uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training --live
#     Or every logged scalar in the browser (http://localhost:6006), also live:
uv run tensorboard --logdir logs/rsl_rl/handle_grab_state

# 4. Camera student (overhead RGB + depth), distilled from the teacher's actor checkpoint. Cameras switch on by
#    themselves. 128 envs, about 3.7 s per iteration and 14 of 16 GB GPU memory on an RTX 5070 Ti; 1600 iterations.
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task=Isaac-HandleGrab-Camera \
    --agent=rsl_rl_distillation_cfg_entry_point --checkpoint=<path/to/teacher/model_XXXX.pt> --viz none
# 4b. Live graphs of the student run (success, grasp rate, student-teacher action loss).
uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training --experiment handle_grab_camera_distillation --live

# Optional: the camera policy trained from scratch instead (slower, the vial task's own comparison).
uv run python scripts/reinforcement_learning/train.py --task=Isaac-HandleGrab-Camera

# Watch a policy (play mode: every episode starts at home, bags in dataset order).
uv run python scripts/reinforcement_learning/play.py --task=Isaac-HandleGrab --checkpoint=<path>
```

Preset: `presets=sim2real`.

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
- `holding` **uses the jaw's closed angle and where the strap sits against the jaws.** With the deformable strap (the
default) that is the strap passing through the gap between the closed jaws; with the rigid strap it is
each jaw's contact
point inside the strap zone. If the closed angle, the jaw points or the zone are wrong, grasps that look right in the
viewer never count, or grasps that are wrong count and get latched into the milestones.
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

Then play the dataset the run wrote, one row at a time (home, pregrasp, grasp, grasp_closing), the way the vial's
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
| Closed mouth (`closed_mouth`, "The grasp box" below)        | Where the strap must cross the closed jaws             | `strap_through_mouth` is 1 for accepted grasp attempts once the jaw closes                                     |
| `ARM_HOME_JOINT_POS`, `BAG_REST_Z`                          | Home pose; where the bag sits                          | `bag_shift_mm`, `bag_lift_mm` and `bag_speed_mps` are about zero before the grasp                               |


Accepted attempts need `peak_lower_force_n` at most 3 and `holding_streak` of 10. If every rejection is `not_holding`
while the viewer shows a good grasp, the zone, the closed-mouth boxes (`closed_mouth` in `scene/assets/jaw_points.json`, see "The grasp box" below)
or the closed-angle constant (`GRIPPER_HOLD_POS` in `mdp/task.py`) are the suspect.

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
| **holding** | The grasp boolean. The jaw is closed (measured `revolute_7` below `GRIPPER_HOLD_POS` = -0.30 rad, within 0.08 rad of the -0.38 rad stop; open is 0.6 rad) **and** the strap crosses the closed mouth from one side of the gripper to the other (the grasp box, below). Needs the deformable strap; there is no contact-based detection. |
| **grasped** | `holding` for 3 consecutive steps. Latches.                                                                                                                          |
| **success** | `holding` for 5 s (150 steps) in a row. Ends the episode. The bag does not have to rise (`LIFT_GRAVEYARD.md`).                                                       |
| **dropped** | After the grasp, not `holding` for 15 consecutive steps (0.5 s). A single lost step is not a drop.                                                                   |

### The grasp box (closed mouth)

The strap counts as held only when it runs through the space the jaws enclose when they are shut, from one side of
the gripper to the other, and only while the jaw is actually shut. An open or half-closed jaw registers nothing,
whatever the strap is doing.

**The shape.** The space inside the closed jaws is a wedge: about 8 mm across at the tips, widening to about 22 mm
near the palm. The palm starts about 22 mm in front of the hinge, where the gap closes to about 3 mm. One box can't
fill a wedge, so the mouth is a stack of four boxes along the jaws, each as wide as the real closed gap in its slice.
The tips' last 20 mm is left out, so a strap held only at the very tips does not count.

| Distance in front of the hinge | Along the hinge | Across (moving-jaw face to rigid-jaw face) |
| ------------------------------ | --------------- | ------------------------------------------ |
| 55-65 mm (nearest the tips)    | 15.0 mm         | 15.4 mm                                    |
| 45-55 mm                       | 16.5 mm         | 18.0 mm                                    |
| 35-45 mm                       | 18.4 mm         | 20.6 mm                                    |
| 25-35 mm (against the palm)    | 21.9 mm         | 21.5 mm                                    |

**How it is built.** `scene/assets/build_handle_assets.py` (`jaw_points()`) reads every vertex of both jaw meshes
from `so101_physics.usda` and turns the moving jaw to its closed stop (-21.7 deg). It then expresses both jaws in a
frame on the hinge: axis 0 along the hinge, axis 1 along the jaws, axis 2 the closing direction. The span along the
jaws, from `MOUTH_BACK` (0.025 m) to `MOUTH_FRONT` (0.065 m), is cut into `MOUTH_SLICES` (4) slices. In each slice the
box spans:
- the hinge width both jaws share,
- across, from the moving jaw's face to the rigid jaw's face.

The boxes are written to `scene/assets/jaw_points.json` as `closed_mouth`: one origin, one set of axes, and a `min` and
`max` per box. The frame is fixed to the rigid jaw (`wrist_roll_08c`), so the boxes move with the hand and do not
follow the moving jaw. That is why the closed-jaw check is needed. The older single `closed_gap` box in the same file
(18.9 x 50.4 x 6.6 mm, tips only) is kept because the direct task, `Isaac-HandleGrab-Direct`, still uses it.

**Where the code is:**

| What                                     | Where                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ |
| The grasp boolean                        | `holding()` in `mdp/task.py`: `gripper_closed(env) & strap_grasp(env).through`              |
| "Closed" threshold                       | `GRIPPER_HOLD_POS` and `gripper_closed()` in `mdp/task.py`                                  |
| Strap against the boxes                  | `strap_in_mouth()` (the test) and `strap_grasp()` (reads the scene) in `mdp/task.py`; returns `StrapGrasp`: `through`, `along`, `back_dist` |
| The two placement rewards                | `grasp_centered()` and `strap_depth()` in `mdp/rewards.py`; weights in `RewardsCfg` in `env_cfg.py` |
| The back point                           | `JawGap.back_point` in `mdp/task.py`                                                        |
| Box data type and loader                 | `JawGap` and `JawGap.from_json()` in `mdp/task.py`; loaded once in `_strap_geometry()`     |
| Box sizes and placement                  | `MOUTH_FRONT`, `MOUTH_BACK`, `MOUTH_SLICES` and `jaw_points()` in `scene/assets/build_handle_assets.py` |
| The generated boxes                      | `closed_mouth` in `scene/assets/jaw_points.json`                                            |
| Grasped / success / dropped from holding | `GrabProgress.update()` in `mdp/progress.py`; the 5 s hold is `HOLD_SECONDS` in `mdp/terminations.py` |

**Side to side, not front to back.** In the box frame, axis 0 runs along the hinge (the gripper's two sides), axis 1
along the jaws (front at the tips, back at the palm), and axis 2 across the jaws (the two jaw faces).
`strap_in_mouth()` tests each column of the strap ribbon as a polyline along the strap. A column crosses a box when:
- it goes from beyond one hinge face (axis 0 below the box's min) to beyond the other (above its max), and
- every part of it between those two faces stays within the box's axis-1 and axis-2 bounds.

So the strap enters through one side of the gripper and leaves through the other. A strap that leaves through the
front (the tips), the back, or a jaw face does not count, and neither does one running along the jaws from front to
back. Any number of strap nodes may lie between the two faces, including none, so a strap crossing a box between two
nodes still counts. The grasp is true when any column crosses any box and the jaw is closed.

`strap_in_mouth()` also returns, for the placement rewards:
- `along`: where along the strap the crossing is, 0 and 1 at the two feet and 0.5 at the top of the arch (NaN without
  a crossing);
- `back_dist`: the distance from the mouth's back point to the nearest point on the strap. The back point is the
  centre of the back face of the box nearest the palm, 25 mm in front of the hinge (`JawGap.back_point`).

Every term that pays for the grasp goes through `holding()`: `held_goal`, the `milestones` grasp, the success
termination, `grasp_centered` and `strap_depth`. So none of them pay while the jaw is open.

The direct task (`Isaac-HandleGrab-Direct`) keeps its own looser test, `strap_through_gap()` (any strap segment
touching its single `closed_gap` box), which this task no longer uses.

**Changing it.** Edit `MOUTH_FRONT`, `MOUTH_BACK` or `MOUTH_SLICES` (and `GRIPPER_HOLD_POS` for the closed angle).
Rerun the build script, which prints one `[jaw_points] mouth ...` line per box, and check the result in the viewer:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/assets/build_handle_assets.py
uv run python -m isaaclab_tasks.core.official_handle_grab.scene.gap_viewer --viz kit
```

`scene/gap_viewer.py` loads a grasp row from the reset dataset, closes the jaw, and draws the boxes on the arm:
- **grey** while the jaw does not count as closed,
- **red** when it is closed without the strap in a box,
- **green** while `holding` is true.

Blue and orange spheres mark the rigid and moving jaw sample points. It prints the jaw angle, the closed flag, whether
the strap is in a box, and `holding` once a second. Use `--jaw open` to hold the jaw open, and `--row N` or `--phase` to
pick another row. Changing the boxes or the closed angle changes the reward, so retrain afterwards.

Actions are relative joint steps: 0.033 rad on joints 1 to 6, 0.04 rad on the jaw (`ARM_ACTION_SCALE`,
`GRIPPER_ACTION_SCALE` in `env_cfg.py`; the vial uses 0.02 on the jaw, see "Known issue"). A zero action holds still.
The sim runs at 60 Hz, the policy at 30 Hz, episodes are 20 s.

## Rewards

All terms are set in `RewardsCfg` in `env_cfg.py`. The functions are in `mdp/rewards.py` unless marked as stock Isaac
Lab terms (`isaaclab.envs.mdp`).

Isaac Lab's reward manager adds `value x weight x step_dt` every policy step, with `step_dt` = 1/30 s. So a term worth
1 with weight 0.1, held for the whole 5 s hold (150 steps), adds 0.1 x 5 = 0.5 to the episode return. The "most per
episode" column uses that rule. "Holding" means the grasp boolean `holding()` is true: jaw closed and strap across the
mouth side to side (see "The grasp box").

### Rewards (positive)

| Term                | Weight | Value each step                                                                                                                 | Pays when                                          | Most per episode                      |
| ------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------- |
| `approach_progress` | 1      | Decrease in jaw-tip-to-strap distance since the last step, in cm, clipped to [-1, 1]. Moving away costs the same.                 | Until the grasp latches; nothing on the first step | About 0.8 for the 25 cm from home     |
| `jaw_closing`       | 10     | Rise in the episode's best `closed fraction x (1 - tanh(tip-to-strap distance / 0.02 m))`; 0 when it does not rise              | Any step; nothing on the first step                | 0.33 for one full close at the strap  |
| `held_goal`         | 0.2    | 1                                                                                                                               | Grasped (latched) and holding                      | 1.0 over the 5 s hold                 |
| `grasp_centered`    | 0.1    | `1 - abs(2 x along - 1)`: 1 at the top of the strap's arch (`along` = 0.5), 0 at either foot, linear in between                | Holding                                            | 0.5 over the 5 s hold at the very top |
| `strap_depth`       | 0.05   | `1 - tanh(back_dist / 0.02 m)`: about 0.75 at 5 mm from the back point, 0.54 at 1 cm, 0.04 at 4 cm                         | Holding                                            | Up to 0.25 over the 5 s hold          |
| `milestones`        | 20     | 1 on the step the grasp latches (3 consecutive holding steps)                                                                   | Once per episode; not on an episode's first step   | 0.67                                  |
| `success`           | 200    | 1 on the step the 5 s hold completes; the episode then ends                                                                     | Once                                               | 6.67                                  |

What each one is for:
- **`approach_progress`** is a difference of potentials. It gives the same signal 25 cm from the strap as 2 cm away
  and pays nothing for hovering, which is what lets a camera policy find the approach from home. The strap point is
  the recorded pinch moved with the bag (`strap_target_w()` in `mdp/task.py`).
- **`jaw_closing`** bridges reaching the strap and holding it; nothing else pays for the 0.9 rad close. Only a new
  episode best pays, so closing at the strap, backing off, opening and coming back earns nothing twice, and closing
  away from the strap earns nearly nothing. The closed fraction is `gripper_closed_fraction()` in `mdp/task.py`.
- **`held_goal`** is the dense signal through the 5 s hold, between the grasp milestone and the success bonus.
- **`grasp_centered`** prefers a grasp on top of the handle over one on its side. `along` is where the strap crosses
  the mouth as a fraction of its length, from `strap_in_mouth()` in `mdp/task.py`.
- **`strap_depth`** prefers a strap deep in the jaws, which is held more securely, over one near the tips.
  `back_dist` is the distance from the mouth's back point (25 mm in front of the hinge, against the palm,
  `JawGap.back_point`) to the nearest point on the strap. `std` (0.02 m) is a term parameter in `env_cfg.py`.
- **`milestones`** and **`success`** are the sparse goals. The grasped flag is an observation, so the once-per-episode
  payment is Markov.

`grasp_centered` and `strap_depth` are gated on holding, so neither pays for a strap that is only near the jaws. They
are small next to `held_goal` and `success`: they decide which grasp the policy settles on, not whether it grasps.

The grasp itself (`holding` true: `held_goal` plus `milestones`, up to 1.67 per episode) is the largest reward except
`success`, which is the same grasp kept for 5 s. It was raised from 0.1 and 10 (0.83) so that it clearly out-pays the
approach (about 0.8) and every placement and closing term.

### Penalties (negative)

| Term                 | Weight     | Value each step                                                                                             | Defined in                                     | Size                                           |
| -------------------- | ---------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| `bag_lost`           | -50        | 1 on the step `bag_dropped` or `bag_out_of_bounds` ends the episode                                         | stock `is_terminated_term`                     | -1.67, once                                    |
| `lower_body_contact` | 0 then -4  | `(force - 1 N) / 5 N`, clipped to [0, 1]; `force` is the arm's contact force on the bag body below the rim   | `lower_body_contact()`, `lower_body_force()` in `mdp/task.py` | Up to -0.13 per step at full weight |
| `action_rate`        | -0.002     | Sum of squared changes in the action since the last step                                                     | stock `action_rate_l2`                         | Small; smoothness                              |
| `joint_velocity`     | -0.0002    | Sum of squared joint velocities [rad^2/s^2]                                                                  | stock `joint_vel_l2`                           | Small; smoothness                              |
| `joint_limits`       | -1         | How far joints 1 to 6 are past their soft limits [rad], summed                                               | stock `joint_pos_limits`                       | Zero inside the limits                         |

What each one is for:
- **`bag_lost`** charges for dropping the strap after the grasp (out of the jaws for 15 steps, 0.5 s) or knocking the
  bag out of the tray. Both also end the episode, which costs the rest of the hold.
- **`lower_body_contact`** keeps the arm off the bag's body. The first 1 N is free, since the scripted demo this task
  replaces brushed the body at about that. Contact with the strap and the top rim is free, since the jaws have to sit
  there. The weight starts at 0 and the `lower_body_ramp` curriculum (`CurriculumCfg` in `env_cfg.py`,
  `linear_weight_ramp()` in `mdp/curriculums.py`) grows it linearly to -4 (`LOWER_BODY_FINAL_WEIGHT`) over 16000
  environment steps (`LOWER_BODY_RAMP_STEPS`, about 250 PPO iterations). At full weight from step 0 the policy learned
  to avoid the bag altogether. This is the only use of the contact sensor; the grasp itself is detected geometrically.
- **`action_rate`** and **`joint_velocity`** are light regularization for smooth motion.
- **`joint_limits`** covers joints 1 to 6 only. The jaw is meant to sit against its stop when it closes.

### Changing them

Weights and term parameters are in `RewardsCfg` in `env_cfg.py`. They can also be overridden from the command line,
for example `env.rewards.strap_depth.weight=0.1`. The grasp boolean the gated rewards read is described in "The grasp
box". Changing a weight changes what the policy learns, so retrain afterwards; the observation layout does not change.

Terminations: `success`, `bag_dropped`, `bag_out_of_bounds` (outside the tray), `unstable_robot`, `time_out`.
The vial task's rack-specific terms are not used.

## Observations

- **Teacher actor** (`policy`): joint position, velocity, target, previous action, end-effector pose, bag pose, strap
point in the base frame, distance/bag-speed/jaw-closed features, the grasp milestone flag.
- **Critic**: the same plus the lower-body force (privileged).
- **Camera student**: two stacked overhead 256 x 144 RGB frames (`overhead_rgb`, uniform noise), two stacked depth
frames (`overhead_depth`), and noisy proprioception. RGB and depth each get their own CNN. The teacher group
(`teacher_state`) is identical to `policy`, so the teacher checkpoint's input layout matches.

### The overhead camera (student)

All in `camera_env_cfg.py`; the state task has no camera, and the direct task keeps the base scene's D415.

- **Pose.** 0.70 m above the tray, looking at the bag's spawn point, tilted about 12 degrees in from +x
(`STUDENT_CAMERA_POS`, `STUDENT_CAMERA_TARGET`, `_look_at_ros()`). Image right is world -x, so the arm is at the top
of the image and the tray sits square in the frame. The view is nearly top-down on purpose: the bag spawns at any yaw,
and a steep side view would hide a strap on the bag's far side. Straight above the bag (the old pose), the wrist
covered the strap at the grasp, because the arm reaches in along +y under the camera; the small tilt from +x sees the
side of the jaw past it.
- **Images.** 256 x 144 at the D415's 69 degree field of view; the strap is about 2 px wide. RGB is normalized per
image. Depth is `distance_to_image_plane` clipped at 1 m (`STUDENT_DEPTH_MAX_M`) and divided by it, so the scene sits
at about 0.3 to 0.86. Pixels with no return read 0, as on the real D415.
- **Lighting.** A dome light (700) for even, shadow-free fill, and a distant key light (1800, 25 degree angle, so shadow
edges are soft) shining from 35 degrees off vertical over the far (+y) end of the tray, onto the front of the jaw.
Off the camera axis, its reflection on the white floor misses the lens. RGB peaks at about 238 of 255, so nothing clips.
This is the default look; `presets=sim2real` varies it (see "Sim to real").

## Sim to real

Isaac ROS is not needed. The trained student is a 30 Hz loop: one D415 frame and seven joint angles in, seven joint
targets out. `real/` runs that loop in plain Python with `pyrealsense2`, the Feetech servo SDK and ONNX Runtime. ROS
would only add a transport layer between one camera and one bus. Run every command below from the Isaac Lab root.

The trained student is `logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25` (`exported/policy.onnx`, with
`policy.onnx.data` beside it). Both files are committed, so a checkout of this branch has them. The teacher it was
distilled from is `logs/rsl_rl/handle_grab_state/2026-11-10_16-22-41/model_400.pt` (not committed; only needed to
retrain).

### 1. Hardware

- **Arm.** USB servo bus, usually `/dev/ttyACM0`. The seven motors map in order to `revolute_1` .. `revolute_7`,
gripper last. Give your user the port with `sudo usermod -aG dialout $USER`, then log out and back in.
`real/joint_map.json` (written by calibration, not committed) stores each motor's id, its encoder count at the
simulated home pose, and its direction.
- **Camera.** A D415 about 0.70 m above the tray, looking down at the bag's spawn point, tilted about 12 degrees in
from +x so the arm is at the top of the image (see "The overhead camera (student)"). Get within about 2 cm and
3 degrees; the twin check below confirms it. Plug it into a USB 3 port. The real 1280 x 720 frame is cropped to the
simulated 69 degree view and resized to 256 x 144 (`bridge.py`, `to_student_frames`). If the real camera sees less
than the simulated one, the bridge refuses to run rather than stretch the image.
- **Scene.** Same tray and bag geometry. Colours and lighting may differ. Even indoor light: no direct sun, and no
glare on the floor.

### 2. Drivers

```bash
uv pip install feetech-servo-sdk pyrealsense2 onnxruntime
```

### 3. Retrain with `presets=sim2real` (recommended)

The student above has only seen one clean rendered look. Retrain so the real camera, room light and object colours are
another sample of what it trained on. The teacher continues from its checkpoint with physics variation (jaw and bag
friction, motor gains, joint friction and armature, encoder bias) and home-only starts. The student is then distilled
from that new teacher with image augmentation, light and colour changes, and camera-pose jitter.

```bash
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task=Isaac-HandleGrab \
    --checkpoint=logs/rsl_rl/handle_grab_state/2026-11-10_16-22-41/model_400.pt presets=sim2real --viz none
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task=Isaac-HandleGrab-Camera \
    --agent=rsl_rl_distillation_cfg_entry_point --checkpoint=<new teacher run>/model_XXXX.pt presets=sim2real --viz none
# The bridge must build exactly the inputs the env does, and ONNX must match policy.pt.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.check_bridge --run <student run> --viz none
```

For a first try on the arm, skip the retrain and use the student run named above.

### 4. On the robot, in order

```bash
# a. Torque off. Pose the arm by hand at the simulated home, then press Enter.
#    Writes real/joint_map.json, marked unverified.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.calibrate --port /dev/ttyACM0 --ids 1 2 3 4 5 6 7
# b. The simulated arm mirrors the real one, and the real camera is overlaid on the simulated view.
#    Move each real joint by hand. Flip "sign" in joint_map.json for any joint that moves the wrong way.
#    When every joint matches and the overlay lines up, set "verified": true.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.twin --camera --viz kit
# c. Drive to home slowly.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.home
# d. Dry run: camera and joints are read, the policy runs, targets are printed, nothing is written.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.deploy \
    --policy logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx
# e. Live: one 20 s episode, step log saved. Point --policy at the retrained student if you ran step 3.
uv run python -m isaaclab_tasks.core.official_handle_grab.real.deploy \
    --policy logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx \
    --execute --log episode.npz
```

Keep `policy.onnx.data` next to `policy.onnx`; the weights are in it.

### 5. Safety

`deploy` will not start if the joint map is not marked verified, if the arm is more than 0.15 rad from home, or if a
motor is above 65 C. While it runs, torque is capped at 60% on the arm and 30% on the gripper. It stops and holds the
arm where it is when a joint lags its target by more than 0.35 rad for 5 steps, when the loop overruns 10 times, or on
Ctrl-C. Keep a hand on the power switch for the first runs.

### 6. After the first runs

Each action is added to the measured position and re-applied at 60 Hz, twice per policy step, as in the simulator.
The simulated arm has gravity switched off, so compare `target` with `joint_pos` in `episode.npz`. A joint that sags
under the bag needs a higher `--arm_torque` or retuned gains in the simulator. The arm's own colour is not randomized,
so a real arm that looks very different from the rendered one can throw the camera policy off.



## Reset dataset

Four phases, 128 rows each (`reset/curriculum.py`): **home** (arm at home, jaw open, bag at rest), **pregrasp** (tip
6 cm back along the approach axis, jaw open), **grasp** (tip on the recorded pinch, jaw open), **grasp_closing** (jaw
slid onto the strap and closed to -0.10 rad, 0.2 rad short of the hold angle, so not yet `holding`; bag still resting).
Lifting is not a phase.

Training starts with the four phases equally likely. The `reset_fade` curriculum term (`fade_reset_phase()` in
`mdp/curriculums.py`, `CLOSING_FADE_STEPS` in `env_cfg.py`) lowers the weight of `grasp_closing` linearly from 1 to 0
over the first 300 PPO iterations, after which training samples home, pregrasp and grasp equally. Evaluation and play
start from **home** only, walking the home rows in order.

The generator places the bag, solves IK for the recorded pinch (`scene/grasp_goal.json`), and drives the arm **in PhysX
through the task's own action interface**. Each phase continues from accepted rows of the one before it. An attempt is
kept only with peak lower-body force of at most 3 N, the bag moved at most 3 cm, and the bag nearly still. Each
rejection reason is counted and printed. Rows store measured joints, the bag pose in the environment frame, and the
strap's shape.

**grasp_closing** (`collect_grasp_closing()` in `reset/generator.py`) starts from a grasp row, slides the jaw
`--close_insert` (15 mm) further along its insert axis by IK, and closes it to `CLOSING_JAW_POS` (-0.10 rad). That state
is the row. Then the generator closes the jaw the rest of the way for `CLOSE_STEPS` (150 steps) and keeps the row only
if `holding` is true for at least the last 15 steps, so every row can be grasped; a row that is already `holding` is
rejected. The slide is needed because the open-jaw grasp leaves the strap at the jaw tips, about 70 mm in front of the
hinge and past the 65 mm front of the mouth: closed there, only about a quarter of attempts held. With the slide about
85% hold, and the strap sits about 30 mm from the back point instead of 44 mm.

**Spawn box.** Defaults are +-3 cm in x, +-4 cm in y and +-20 degrees of yaw around the scene's nominal spot
(`--x_half --y_half --yaw_half_deg`). The nominal spot is only 14 mm from the tray's near wall, so poses whose rotated
footprint would come within 5 mm of a wall are resampled. Effectively y covers about -3.9 cm to +1 cm around the
nominal, not the full +-4 cm. Widen it toward +y only if the arm can still reach (the handle must stay within about
0.30 m of the base).

## Known issue: the teacher never closed the jaw (fixed 2026-11-10)

**Symptom.** The teacher trained on 2026-11-10 (`logs/rsl_rl/handle_grab_state/2026-11-10_13-58-15`, stopped at
iteration 850) moved the jaw to the strap, hooked it on one jaw, and waited out the 20 s with the jaw open.
`Metrics/grasp_rate`, `Metrics/success_rate`, and `Episode_Reward/` `held_goal`, `grasp_centered`, `strap_depth`,
`milestones` and `success` were 0 at every iteration, and every episode ended in `time_out`. `approach_progress` was
the only reward it ever earned. `Policy/mean_std` fell from 0.20 to 0.077.

**What was checked and was fine.** The grasp boolean and the rewards: from grasp rows with the jaw commanded shut, 28 of
32 environments held the strap and every holding reward paid. The exploration noise matched the vial (Gaussian on all
seven actions, `init_std` 0.2, `std_range` 0.05 to 0.3); play runs the mean action, which is why the joints look
noiseless there.

**Cause.** The jaw has to travel about 0.9 rad from open (0.6) to the hold angle (-0.30). At 0.02 rad per step that is
45 full-scale steps in a row, and with 0.2 noise the jaw wanders about 0.1 rad in a whole episode. No reward paid for
closing before `holding` was already true, and no episode started holding, so the policy never saw what a grasp is
worth. The vial gets away with the same noise because its dataset has closed-grasp rows; this one had dropped them with
lift.

**Why the grasp reward alone did not close the jaw.** It would, if the policy reached a grasp sometimes. Reinforcement
learning only learns from rewards it collects, and in 850 iterations (about 55 million steps) there were no grasps, so
every grasp reward stayed at 0. The grasp terms (`held_goal`, `milestones`, `success`, `grasp_centered`,
`strap_depth`) all need `holding` to be true already. They are all-or-nothing: a jaw 50% or 90% closed earns the same as
an open one, so partial closing gives nothing to follow. Each fix covers a different part:

- **Closed-grasp starts** make the grasp reward experienced at all. The critic learns that holding is worth a lot (the
  success bonus alone is 6.67), but these episodes start fully closed, so mostly they teach "stay closed once holding",
  not what to do halfway through the close.
- **`jaw_closing`** is the gradient from open to the point where the grasp rewards take over. It is kept small (at most
  0.33 per episode, against about 8.3 for a grasp held to success) so "close near the strap and stop" does not pay.
- **The larger jaw step** makes the close twice as easy to stumble into.

With closed-grasp starts the grasp reward might be enough without `jaw_closing`. The vial has no closing reward, but
its pregrasp jaw is already half closed, so its gap is smaller. To test that, train once with
`env.rewards.jaw_closing.weight=0` and compare `Metrics/grasp_rate` against a run with the term on.

**Fixes.**

| Fix                         | Where                                                                    | What                                                                                     |
| --------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Closed-grasp reset phase    | `reset/curriculum.py`, `reset/generator.py`                              | A quarter of training episodes started holding the strap. Replaced the same day, see below. |
| Closing reward              | `jaw_closing` in `mdp/rewards.py`, weight 10 in `env_cfg.py`            | Pays closing the jaw at the strap, once per episode (see "Rewards").                     |
| Larger jaw step             | `GRIPPER_ACTION_SCALE` in `env_cfg.py`, 0.02 to 0.04 rad                 | Half as many steps to close, twice the exploration on the jaw. Also doubles the squeeze. |

Check after the fixes, in the training environment: from `grasp_closed` rows all 32 environments hold, hold for the
full 5 s and pay `success`; from `grasp` rows with the jaw commanded shut, `jaw_closing` pays 0.22 and 14 of 32 hold
(the faster jaw pushes the strap out more often than the old one did, which the closed-grasp starts teach the policy to
avoid by going deeper first).

### Follow-up: closed-grasp starts handed out free successes (fixed 2026-11-10)

**Symptom.** The next run (`2026-11-10_15-16-00`) reported 96% `Episode_Termination/success` by iteration 120. But a
quarter of its episodes started from `grasp_closed`, already holding the strap, and those only had to not let go for
5 s to earn the full grasp and success rewards. The training success rate overstated how well the policy grasps from
home, and these starts taught "stay still" more than "grasp".

**Fixes.**

| Fix                         | Where                                                                              | What                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Nearly closed starts        | `GRASP_CLOSING` in `reset/curriculum.py`, `collect_grasp_closing()` in `reset/generator.py` | `grasp_closing` replaces `grasp_closed`: the jaw is 0.2 rad short of holding, so the policy has to finish the close to earn anything. Dataset version 4. |
| Fade them out               | `reset_fade` in `CurriculumCfg` (`env_cfg.py`), `fade_reset_phase()` in `mdp/curriculums.py` | Their share falls from a quarter to none over the first 300 iterations, so later grasps start further out. |
| Success per start type      | `success_by_start` in `CurriculumCfg`, `success_by_start` in `mdp/curriculums.py`  | Logs `Curriculum/success_by_start/<phase>`, a running success rate per start; `home` is the number play measures. Plotted by `plot_training.py`. |

Check in the training environment: from `grasp_closing` rows with the arm and jaw held still, 0 of 32 hold and no grasp
reward pays; with the jaw commanded shut, 32 of 32 hold and `held_goal`, `milestones`, `grasp_centered`, `strap_depth`
and `jaw_closing` pay.

The dataset has to be regenerated after any change to the closing phase.

## Evaluation

`evaluation.evaluate(env, policy, episodes_per_env=2)` runs complete episodes from home (build the env from
`resolve_task_config(..., play_mode=True)`) and returns grasp, lift, success, drop and out-of-bounds rates, mean time
to success and lower-body force. Training logs `Metrics/grasp_rate`, `Metrics/lift_rate`, `Metrics/success_rate` and
`Safety/peak_lower_body_force_n`, but those average over all four reset phases and overstate the home task.
`Curriculum/success_by_start/home` is the training-time success rate of episodes that started at home alone.

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
- The camera is a fixed overhead D415 with RGB and depth, not a wrist camera with RGB only. It is a regular `Camera`,
not a tiled camera; with 128 envs at 256 x 144, collection takes about 2.7 s per distillation iteration.



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

Visualizing the workflow 

```bash
B2 scene ──► B3 name it + prove it ──► B4 record the pinch ──► B5 check the motion
   (data)      (lookup + tests)          (grasp_goal.json)       (open-loop physics)
                                                │
          B6 the task's rules ◄─────────────────┘   (rewards and IK aim at that pinch)
                │
          B7 generate start states ──► B8 smoke test ──► B9 teacher ──► B10 student ──► B11 evaluate
```



### B1. Every file

Paths are under `source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/` unless they start with `core/` (relative to `source/isaaclab_tasks/isaaclab_tasks/`) or `source/`.

From first principles we can summarize our files as: 

1. World: scene/, scene/assets
2. RUles: env_cfg.py, mdp/, reset/
3. LEarning: agents/, camera_env_cfg.py, evaluation. py [e](http://evaluation.py)


| Our file                                                            | Vial counterpart           | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                                                       | `config/so101/__init__.py` | Registers `Isaac-HandleGrab` and `Isaac-HandleGrab-Camera`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `env_cfg.py`                                                        | `env_cfg.py`               | The state task: scene, actions, observations, events, rewards, terminations, curriculum, `sim2real` and `pinch_only` presets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `camera_env_cfg.py`                                                 | `camera_env_cfg.py`        | The overhead-camera task (D415, RGB and depth).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `agents/rsl_rl_ppo_cfg.py`                                          | same                       | PPO for the teacher (1500 iterations) and a from-scratch camera policy (2500).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `agents/rsl_rl_distillation_cfg.py`                                 | same                       | Distillation for the camera student (1600 iterations), with `BoundedTeacherDistillation`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `mdp/task.py`                                                       | `mdp/task.py`              | The grasp boolean `holding` and the closed-mouth test (`GRIPPER_HOLD_POS`, `JawGap`, `strap_in_mouth`, `strap_grasp`; see "The grasp box"), the lower-body force, the tip and strap positions, `ARM_BASE_ROT`, and the loader for `grasp_goal.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
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
| `scene/gap_viewer.py`                                              | none                       | Draws the closed-mouth grasp boxes on the arm and prints the holding test (see "The grasp box").                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `scene/grasp_pose_tool.py`                                          | none                       | Records those files. Live mode reloads `scene/grasp_poses.json` when you save it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `source/isaaclab_tasks/test/core/test_official_handle_grab.py`      | `tests/`                   | All our tests, in one file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |




The real-robot half is `real/` ("Sim to real"): `calibrate.py`, `twin.py` and `home.py` stand in for the vial's
`utils/calibrate.py`, `inspect_so101.py` and `home_so101.py`; `deploy.py` runs the exported ONNX student instead of a
LEAPP bundle; `check_bridge.py` has no vial counterpart.

### Josh Edit B2. Step 0: Setting the scene

The vial puts the spawned world on one class, `SO101SceneCfg`, and the file paths in `scene/assets/__init__.py` 

Ours is the same split. Paths are in `scene/assets/__init__.py`. 

Inside /scene/assets/**init**.py we define: 

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

We view the scene we just built, frozen: no recorded motion, no joint noise, no action noise, and the bag at its
nominal spot. The direct task replays `scene/grasp_poses.json` by default; we turn that off here because the motion
is recorded in B4 and replayed in B5.

```bash
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit \
  env.use_demo_residual=false \
  env.arm_joint_noise=0 \
  env.bag_x_range=[0,0] \
  env.bag_y_range=[0,0] \
  env.bag_yaw_range=[0,0] \
  env.action_noise_model=null
```

Note: The direct task / student trainer builds its own scene from the same asset configs and adds the D415. The manager scene leaves the camera off, the way the vial adds its camera in `camera_env_cfg`.

### Josh Edit B3: Name the task and check numbers

handle_grab_[scene.py](http://scene.py) does not run a robot. It stores numbers and file paths: the tray USD, the arm USD, the bag USD, the base position (0, -0.20, 0.06), and the 180 degree yaw. env_[cfg.py](http://cfg.py) wraps that scene in a larger description called HandleGrabEnvCfg: how big a joint step is, what counts as success, what the reward pays. That object is still just data in memory.

We can look at the environment configuration in
`source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/env_cfg.py`.

**To be more specific, our environment config file has ten different congig classes:** 

ActionsCfg — relative joint targets for the arm and the gripper.

PolicyStateGroupCfg — actor observations (joints, last action, end-effector, bag, contact).

CriticStateGroupCfg — the same observations, plus anything the critic sees extra.

ObservationsCfg — bundles the policy and critic groups.

EventsCfg — reset randomization: bag friction, arm gains, and dataset start poses.

Sim2RealEventsCfg — that same reset, plus jaw friction, gripper gains, joint parameters, and encoder bias for robot transfer.

RewardsCfg — approach and hold shaping, grasp/lift milestones, success bonus, drop penalty, and small regularization terms.

TerminationsCfg — success, bag dropped, bag out of bounds, unstable robot, and timeout.

CurriculumCfg — ramps up the lower-body contact penalty over training.

HandleGrabEnvCfg — the environment: scene, actions, observations, events, rewards, terminations, and curriculum.

**We can think of this step as three separate ideas.** 

1. ++Config as data - we need a name to point at it++

In the last step we only defined the world. In other words, we only have a python obect that *describes* a world - nothing can ask for it yet. 

gym.register in official_handle_grab/**init**.py writes one line into a phone book. For instance, if we write: 

gym.register(id="Isaac-HandleGrab", ...) in official_handle_grab/**init**.py, we get

"Isaac_HandleGrab" -> build env_cfg:HandleGrabEnvCfg, run with ManagerBasedRLEnv, train it with HandleGrabStatePPORUnnerCfg

So isaalab_tasks/**init**.py  imports every task package automatically. In other words, when we run import isaaclab_task , our phone book is automatically filled. 

1. ++What/who looks up a name and when?++

Every later step starts by asking for a specific task by a specific name. THey all go through the same lookup: 

resolve_task_config(name, ...) 

this returns the configuration: 


| Program                | Section | What it does with the world                         |
| ---------------------- | ------- | --------------------------------------------------- |
| pose tool              | B4–B5   | Moves the arm by hand-written joint angles          |
| generator              | B5, B7  | Solves IK and keeps the grasps that physically work |
| `train.py` / `play.py` | B9–B11  | Runs thousands of copies with a policy              |


Key: the policy never sees handle_grab_scene.py - it only sees whatever the name resolves to. 

1. ++The lookup can succeed and still be wrong: test it without Kit++

A lookup that works can still hand back the wrong world. If we point to an old scene class, for instance, Kit would
spawn the wrong world without any error.

B3 needs no recorded data. None of these tests read `scene/grasp_poses.json` or `scene/grasp_goal.json`; when
`grasp_goal.json` is missing, `load_grasp_goal()` in `mdp/task.py` falls back to a nominal pinch point, so the configs still build. Recording starts in B4. The nominal pinch point is a hard-coded constant in [mdp.task.py](http://mdp.task.py) - we coded this to be at the top of the handle/arch with the jaw tip pushed down past the strap's bottom edge. 

We have three tests, and each catches one kind of mistake. Run them from the repository root.

**Test 1.** `test_the_arm_stands_upright_on_the_tray_wall`**: the name gives us the world we built**

- Resolves `"Isaac-HandleGrab"` (turns the string into the config object in our phone book).
- Composes the spawn rotation of the SO-101, because the original URDF spawns the arm lying down.
- Checks that the arm's +Y axis comes out along the world's +Z axis (arm standing up).
- Checks that the position, rotation, and both USD paths equal the constants in the scene file.

If the registration is broken, it fails before comparing anything. The test never opens Isaac Sim: it looks up the
name, gets `HandleGrabEnvCfg` back, and compares fields on that object with the constants in `handle_grab_scene.py`.

```bash
uv run python -m pytest source/isaaclab_tasks/test/core/test_official_handle_grab.py::test_the_arm_stands_upright_on_the_tray_wall -q
```

Check: `1 passed`.

**Test 2.** `test_lift_and_success_need_the_strap_in_the_jaws_not_just_a_rising_bag`**: success means a held bag**

- Feeds `GrabProgress` 12 fake steps for 2 envs. Both bags rise 2 cm, but only env 0 is holding.
- Only env 0 earns grasp, lift and success, and its time to success is step 7.

A view can show a bag in the air, but it can't show whether the task would count that as a success. This test does.


| **What**                                                                                                                                                                                       | **Path**              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| The rule this test calls: lift latches only when the bag has risen and the jaws are holding it; success only counts while the hold continues. A rising bag with holding = false earns nothing. | `mdp/progress.py`     |
| Live thresholds and the termination that calls the rule (when to end the episode)                                                                                                              | `mdp/terminations.py` |
| What "holding" means in the simulator (jaw closed, both jaws on the strap)                                                                                                                     | `mdp/task.py`         |
| Success reward                                                                                                                                                                                 | `mdp/rewards.py`      |


To change the rule, edit `progress.py` (or the thresholds in `terminations.py`), then rerun the test.

```bash
uv run python -m pytest source/isaaclab_tasks/test/core/test_official_handle_grab.py::test_lift_and_success_need_the_strap_in_the_jaws_not_just_a_rising_bag -q
```

Check: `1 passed`.

**Test 3.** `test_env_cfg_no_forbidden_imports.py`**: the configs build before Kit starts**

Training scripts load the config first and launch Isaac Sim second, so building the config must not import
simulator-only modules. This lives in its own file; `-k HandleGrab` picks our three tasks.

```bash
uv run python -m pytest source/isaaclab_tasks/test/core/test_env_cfg_no_forbidden_imports.py -q -k HandleGrab
```

Check: `3 passed` (`Isaac-HandleGrab`, `Isaac-HandleGrab-Camera`, `Isaac-HandleGrab-Direct`).

Optionally, run the whole handle-grab test file (rewards, dataset, IK math, presets). All of it runs without Kit:

```bash
uv run python -m pytest source/isaaclab_tasks/test/core/test_official_handle_grab.py -q
```

**How you know B3 is done:** all three tests pass. Nothing has moved in the simulator yet and nothing is recorded.

### Josh Edit B4: Record one pinch

**Question:** where exactly should the jaw be when it grabs the strap?

B3 checked the names and the success rule but never moved the jaw. Here we hand-pose the arm once, at the bag's
nominal spot, and record that pose. It becomes the target that every later step aims at: the generator's IK (B7)
and the approach reward (B6).

Two files, two roles:

- `scene/grasp_poses.json` is the **input you write**: joint angles in degrees for each waypoint. A waypoint lists
only the joints it changes; the rest are inherited from the waypoint before it.
- `scene/grasp_goal.json` is the **output of** `--record`: the rigid-jaw tip position and orientation in the bag
frame, the gripper angle, and the joint angles.


| Waypoint       | What it should be                                                      |
| -------------- | ---------------------------------------------------------------------- |
| `pregrasp`     | Jaws open, a few cm back from the strap                                |
| `grasp`        | Jaws open, straddling the strap. **This is the one you record.**       |
| `grasp_closed` | Same arm pose as `grasp`, gripper closed on the strap (replayed in B5) |
| `lift`         | Arm raised, gripper still closed (optional: training learns the lift)  |


Record `grasp` (jaw open), not `grasp_closed`. The generator takes the open-jaw pose, solves IK for it at each bag
position, and then closes the jaw itself.

**Starting from no data.** Delete `grasp_goal.json` and `grasp_trajectory.json`, but do not delete
`grasp_poses.json`: `Isaac-HandleGrab-Direct` reads it while the environment is built, before the tool could write a
template, and stops with `FileNotFoundError` if it is missing. It also needs a `"sequence"` that contains
`grasp_closed` and `return_home`. Start instead from a file where every waypoint is the home pose:

```json
{
  "show": "grasp",
  "sequence": ["pregrasp", "grasp", "grasp_closed", "lift", "return_home"],
  "pregrasp": {"revolute_1": 6.3, "revolute_2": 53.9, "revolute_3": -124.3, "revolute_4": 40.7,
               "revolute_5": -5.2, "revolute_6": -21.2, "revolute_7": 34.4},
  "grasp": {"revolute_1": 6.3, "revolute_2": 53.9, "revolute_3": -124.3, "revolute_4": 40.7,
            "revolute_5": -5.2, "revolute_6": -21.2, "revolute_7": 34.4},
  "grasp_closed": {},
  "lift": {},
  "return_home": {"revolute_1": 6.3, "revolute_2": 53.9, "revolute_3": -124.3, "revolute_4": 40.7,
                  "revolute_5": -5.2, "revolute_6": -21.2, "revolute_7": 34.4}
}
```

The home angles are `ARM_HOME_JOINT_POS` in `scene/handle_grab_scene.py`, in degrees. An empty waypoint inherits
everything from the one before it.

**The strap is deformable.** `USE_DEFORMABLE_HANDLE = True` in `scene/handle_grab_scene.py` makes the strap a stiff
PhysX surface deformable whose feet are clamped to the bag body: it holds its arch (within about 2 mm) and bends only
where something presses it. Its mesh, `scene/assets/bag_handle_ribbon.usda`, is ignored by git, so build it once on a
new checkout, and again after changing `SPACING` in the script:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/assets/build_handle_assets.py
```

Check: it prints `[ribbon] 66 vertices (22 x 3)`. A different count means `bag_handle_info.json` and the mesh no
longer match the reset dataset, which then has to be regenerated. It also prints `[jaw_points] closed-jaw gap`, the direct task's
single tip box (about 18.9 mm along the hinge x 50.4 mm along the jaws x 6.6 mm across). It then prints four
`[jaw_points] mouth ...` lines, the closed-mouth boxes this task's grasp test uses (25-65 mm from the hinge; see "The
grasp box (closed mouth)").

**The tool has two modes, and each shows a different thing:**

| You're changing | Mode | Picks up edits |
| --- | --- | --- |
| **Positions** (joint angles) | Live: no `--mode` flag | On every save. The arm jumps straight to the `show` waypoint and ignores speeds. |
| **Speeds** (`speed_deg_per_s`) | Run: `--mode run` | Only at launch. Plays the sequence at your speeds, then holds. Relaunch after each edit. |

Get the positions right in live mode first, then switch to run mode to tune and check the speeds.

**Step 1. Pose the arm live.** The tool runs `Isaac-HandleGrab-Direct` with the bag at its nominal spot, no noise
and no demo replay. It holds the `show` waypoint and reloads the file every time you save it:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit
```

Edit the `grasp` block in your editor and save. About once a second the terminal prints two lines:

```text
[grasp_pose] grasp  tip in bag frame (x,y,z)=(...) m | to pinch target ... cm | rigid jaw ... mm from strap | moving jaw ... mm from strap | strap pinched .. nodes ... | gripper ... deg | grasped ...
[grasp_pose] joint target/actual deg   1: target/actual | 2: ... | ...
```

With the deformable strap (the default) each jaw reads its distance to the nearest strap point, `rigid jaw 4.2 mm
from strap`, with `TOUCH` under 6 mm, followed by `strap pinched N nodes`. With the rigid strap
(`USE_DEFORMABLE_HANDLE = False`) each jaw reads `strap` when it touches the strap and `none/body` otherwise.

Checks before recording:

- In the window, the open jaws straddle the strap and the bag body is untouched.
- The jaws sit just either side of the strap: both distances are a few mm to about a centimetre, roughly equal. With
  the jaw open they need not read `TOUCH` yet. Check in the window that neither jaw presses on the bag body, and that
  the strap is not bent (an open jaw that dents it is inside it).
- No `!` on the joint line. A `!` means that joint is more than 3 degrees short of its target: it is blocked by
the tray or bag, or sits at a joint limit.
- Wrist pitch (`revolute_4`) is not near its limit. The generator needs joint headroom to solve IK for other bag
positions; a joint parked at its stop is why IK fails later (`pregrasp_ik_joint_margin_rad` at 0.03).
- Ignore `to pinch target` for now: it measures the distance to the *previous* `grasp_goal.json`, or to a nominal
point if there is none.

Set `pregrasp` too (same pose, backed off a few cm along the approach). `--record` plays the sequence from home
through `pregrasp` to `grasp`, so the path in has to be clear.

**Step 2. Record it.** This plays the sequence up to the `show` waypoint, lets the arm settle for a second, and
writes `scene/grasp_goal.json`:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit --record
```

Checks:

- The terminal prints `RECORDED goal -> .../grasp_goal.json` with `"waypoint": "grasp"`.
- `grasped is False at this pose` is expected for the open-jaw `grasp` waypoint.
- `tip_pos_bag` matches the live readout from Step 1 to within a few mm. If it doesn't, the arm took a different
path in (it hit something during `pregrasp`); fix `pregrasp` and record again.

The joint angles in `grasp_goal.json` are only the seed for the generator's IK. The tip pose and jaw orientation
are the target.

**How you know B4 is done:** `grasp_goal.json` exists and was recorded from an open jaw straddling the strap, with no
blocked joints.

**Vial parallel:** the vial hard-codes `WORKSHOP_`* joint constants from one real-robot episode. Our
`grasp_goal.json` plays that role.

### Josh Edit B5: Play the hand waypoints once

**Question:** does the recorded motion actually close on the strap in physics, with no learning involved?

First fill in `grasp_closed` in `grasp_poses.json`: copy `grasp` and change only the gripper (`revolute_7`) to
closed. Set `"show": "grasp_closed"` to watch it live if you want.

`--mode run` plays the sequence and prints every stage. If the file has no `"sequence"` key, it plays `pregrasp`,
`grasp`, `grasp_closed`, `lift`.

**Step 1. Stop once the jaw has closed on the strap.** This is the check that matters for training:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit --mode run --stop_after grasp_closed
```

Checks:

- The last line reads `stopped after 'grasp_closed': grasped at the end = True`.
- In the `grasp_closed` status line, both jaws read `TOUCH` (`strap` with the rigid strap) and `bag lift` is
  about 0.0 cm (the close did not drag the bag).
- With the deformable strap, the readout shows `strap through closed-jaw gap OK` and `gripper` below 0 deg.
  `Isaac-HandleGrab` uses the same rule for `holding`. The `pinched` and `on rigid jaw` counts are for information.
- No `!` peak-speed warnings on the `->` lines (a joint above the 286.5 deg/s motor cap lags its command).

**Step 2. Save the trajectory** (same replay, every control step written to `scene/grasp_trajectory.json`):

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit --mode run --stop_after grasp_closed --save_traj
```

Check: `grasp_closed: grasped NN% ...` is high (most of the steps after the jaw shuts), and the strap gaps at the
end are small.

**Step 3 (recommended). The full sequence, including a scripted lift.** Fill in `lift` first. In the template it
equals home, so the arm would swing back.

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit --mode run
```

Check: `SUCCESS: bag lifted N cm while pinched`. Training does not replay this lift (PPO learns its own in B9), but
a success here proves the strap and jaw can carry the bag at all, which the generator's gate below also requires.

**Gate before B6: does the generator accept the recorded pinch?** This is the generator's one-attempt check from
"Calibrate before you train", Steps 1 to 3. It loads `grasp_goal.json`, so it only makes sense after B4. The
`--output` path keeps the one-attempt dataset away from the file training reads.

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator \
  --viz kit --num_envs 1 --attempts 1 --max_batches 1 --output /tmp/handle_grab_check.pt
```


| Printed line                                        | Want                                                                                                                   |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `tip at recorded joints`                            | about (-0.001, 0.052, 0.026) for the upright mount. Far off means the mount (`ARM_BASE_ROT`, `ARM_BASE_POS`) is wrong. |
| `seed_tip_err_mm`                                   | below 50 (the generator refuses above 50), ideally well below                                                          |
| `home batch 1: accepted`                            | 1/1                                                                                                                    |
| `pregrasp`, `grasp`, `grasp_closed`, `lift` batches | accepted 1/1 each                                                                                                      |
| `rejected by reason`                                | all zeros                                                                                                              |


The generator proves each closed grasp by raising the wrist 2 cm and requiring the bag to rise 1 cm, then builds the
`lift` rows by raising it a further 5 cm and requiring 3 cm more. Both raises come from IK, not from your `lift`
waypoint, but they fail for the same reasons a scripted lift fails, which is why Step 3 above is worth running.

If the run is accepted, play the rows it wrote:

```bash
uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit --dataset /tmp/handle_grab_check.pt
```

If it is rejected, read the reason and go back to B4:

- `ik_pregrasp_failed` or `ik_grasp_failed` with a small `*_ik_best_pos_err_mm` and `*_joint_margin_rad` at 0.03:
a joint is pinned at its limit. Re-pose `grasp` with more headroom (usually `revolute_4`, wrist pitch).
- A large `*_ik_best_pos_err_mm`: the pose is out of reach. Move the grasp, or the bag, closer to the arm.
- `not_holding` while the viewer shows a good grasp: the strap zone or closed-angle constants are wrong (see
"Calibrate before you train", Step 4).
- `ik_proof_failed` or `ik_lift_failed`: the wrist cannot rise from the grasp pose. Same fix as a pinned joint:
re-pose `grasp` with headroom above it.

**How you know B5 is done:** `grasped at the end = True` after `grasp_closed`, and the generator accepts the
one-attempt check.

### Josh Edit B6: Write the task

Here we learn how physics become numbers a learner can use. 

We're building in this order, because each piece reads the one before:

1. **Measurements (**`mdp/task.py`**):** tip position, strap target, where the strap sits against each jaw, whether the jaw is closed. Out of those comes `holding` = jaw angle below 0 **and** the strap passing through the gap between the closed jaws (rigid strap: both jaws' contact points inside the strap zone).
2. **Memory (**`mdp/progress.py`**:** `GrabProgress` latches the flags once they happen:
  - `grasped`: holding for 3 consecutive steps
  - `lifted`: bag 1 cm above its lowest height while holding
  - `success`: held and lifted for 150 steps
  - `dropped`: after the grasp, not holding for 15 steps
  They live in one shared object so every term sees the same contact reading each step.
3. **The referee (**`grab_success` **termination, listed first):** it updates that memory every step. Isaac Lab computes terminations before rewards, so the rewards read this step’s flags.
4. **The pay (**`mdp/rewards.py`**):**
  - **approach:** pays for *getting closer*, not for *being close*, so hovering earns nothing.
  - **milestones:** paid once, never for a milestone the reset row already had.
  - **success bonus, loss penalty, and small smoothness costs.**
  Isaac Lab multiplies each term by the step time (1/30 s).
5. **What the policy sees (**`mdp/observations.py`**):** state, plus the latched flags. If the policy didn’t see whether it had already grasped, it couldn’t tell why the same pose pays differently. That’s what makes once-only rewards learnable.
6. **Actions:** relative joint steps (0.033 rad on the arm, 0.02 rad on the jaw).

**How you know it’s done:** the B6 tests pass (approach, milestones, presets) and the config resolves. It still can’t run, because there’s no dataset yet.

**Vial parallel:** A5–A6. Your `dropped` rule is the one real behavior difference from the vial (we discussed it last time). Wherever you land, B6 is where to write the choice down.

### Josh Edit B7: Generate rows and watch

**Question:** where do episodes start?

Here the chicken-and-egg resolves. The generator builds `Isaac-HandleGrab` itself, then strips it down (`configure_env_cfg`):

- it swaps the dataset reset for a plain reset to defaults
- it removes the rewards, terminations and curriculum
- it sets 120 s episodes
- it keeps the friction and gain randomization, so the grasps it keeps survive those variations

**What one attempt does:**

1. Places the bag in the spawn box (±3 cm in x, ±4 cm in y, ±20° yaw, away from the walls) and lets it settle. That’s the **home** row.
2. Solves IK for a **pregrasp** 6 cm back, choosing randomly among the valid solutions so the arm poses in the dataset vary.
3. Solves IK for the **grasp**, starting from the pregrasp solution.
4. Drives the arm there through the task’s own action interface, at 0.4 rad/s, and closes the jaw: **grasp_closed**.
5. Keeps the attempt only if, at the end: it was holding for the last 10 steps, peak lower-body force ≤ 3 N, the bag shifted ≤ 3 cm and rose ≤ 5 mm, and the bag is nearly still.

Each accepted attempt gives one row per phase. Each rejection reason is counted and printed, and those counts are what you read to tune the generator. Then `reset/viewer.py --viz kit` plays the result attempt by attempt, home → closed.

**How you know it’s done:** `handle_grab_reset_dataset.pt` exists, with a healthy acceptance rate and rows that look right in the viewer.

**Vial parallel:** A7–A8. The vial has 8 phases with 128 rows each, reached by running the task forward from one real-robot episode. Yours has 4 phases ending at the closed grasp, so learning has to find the lift itself. Also say that the `.pt` file isn’t committed, so every new machine has to regenerate it.

### Josh Edit B8: Smoke-test environment

**Question:** does the real task step without crashing, before you spend GPU hours on it?

**What you do:**

1. Run `zero_agent.py --task=Isaac-HandleGrab`. This is the first time that name runs.
2. Run `benchmarks/runtime.py` for throughput.

**What “done” looks like:** no crash, the four kinds of start state look right, and the arm holds still under zero actions. That last check is the gravity question from last time: with relative actions, a zero action gives no holding torque. If the arm sags here, training will fight that the whole time.

**Vial parallel:** A10.

### Josh Edit B9: Train state teacher

**Question:** can a policy that sees everything solve the task?

**What you do:** `train.py --task=Isaac-HandleGrab`, PPO for 1500 iterations on 1024 envs.

- The **actor** sees the policy observations.
- The **critic** also sees privileged contact information. It only exists during training, so it can use information a real robot wouldn’t have.

**What to watch:**

- `Metrics/success_rate` averages over all four start phases. A quarter of those episodes start already grasped, so the number overstates what happens from home. The honest number comes from B11.
- Also watch `grasp_rate` and `lift_rate`, to see which stage stalls.



### Josh Edit B10: Add Camera and Train the Student

**Question:** can a policy that only sees pixels and its own joints do the same thing?

**What you build:** `camera_env_cfg.py` adds the overhead D415. The student sees two stacked 160×90 RGB frames plus noisy proprioception. `teacher_state` repeats the B9 policy observations exactly, so the teacher checkpoint’s inputs still match.

**How distillation works (DAgger-style):**

- the student drives the arm
- the frozen teacher labels every state the student visits
- the labels are clipped to [−1, 1]

The student learns to copy the teacher in the states it actually reaches.

**Command:** `--agent=rsl_rl_distillation_cfg_entry_point --checkpoint=<teacher>`. I checked that the flag exists.

**Vial parallel:** A9 and A11. The vial uses a wrist camera with lighting and camera geometry randomized each episode.

### Josh Edit B11: Evaluate

**Question:** what is the real success rate on the real task?

**What you do:** build the env in play mode. Every episode then starts at home, the bags come in dataset order, and friction and gains are pinned. Run `evaluation.evaluate(env, policy, episodes_per_env=2)`, which returns grasp, lift, success, drop and out-of-bounds rates, time to success, and lower-body force.

**Vial parallel:** A12. The vial uses a play callback that prints one JSON record per run. Same idea, different packaging.