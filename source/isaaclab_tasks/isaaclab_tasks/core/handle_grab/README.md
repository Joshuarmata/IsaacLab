# Grasp the handle

Find the strap and close on it. This is the step before the bag is lifted. The lift policy and the full
pick-up-and-carry task are not in this folder.

Task id: `Isaac-HandleGrab-Direct`. A zero action replays `grasp_poses.json` through `grasp_closed` and then
holds that pose. The policy learns a residual on top (`action_scale` 0.35 rad). Every episode places the bag
somewhere in the tray and turns it about vertical. The teacher still reaches for the strap where it was recorded.

## Training Steps

1. Grasp the handle of a still bag
2. Grasp the handle of a bag that spawns at a random place and heading
3. Add a camera that spawns at varying z-axis locations (different height configurations) and grasp the handle
  1. Teacher Policy is given pose of all objects
  2. Student Policy is only given camera location relative to base_link, current joint angles, RGB + depth from RealSense
4. Change luminosity and grasp handle
5. Add high-contrast elements and grasp handle

## What succeeded

The grasp is a pinch of the 13 mm strap, one jaw on each side. Threading the loop did not work: a 5000-epoch
run of that target reached grasp rate 0, with the tip stuck at the loop plane.

Reward is only the debounced grasp (+5 while it holds) and contact on the lower 95% of the bag body (−0.5).
The top rim and the handle are not collisions. There is no lift term and no success bonus.

Training stayed on the teacher because of these choices in `agents/rl_games_ppo_cfg.yaml`:

- The action head starts at zero. A fresh policy replays the demo. The default initialization pushed the arm
off the teacher and the grasp rate fell to 0 within 25 epochs.
- Exploration is fixed at log-std −1.5 (std about 0.22), about 4.5 deg on the teacher. More noise knocks the
strap out of the jaws. A std of 0 made the KL explode.
- Learning rate is adaptive, starting at 2e-4, with `kl_threshold` 0.01. A constant 3e-4 reached about 75%
and then overwrote it.
- `stagger_first_episodes` spreads the envs across the demo. Without it every env reset together, each epoch
saw one slice of the demo, the KL spiked, and the adaptive rate froze near 1e-6. That was the
solved-then-forgotten run.
- `gamma` 0.997 is about an 11 s horizon at 30 Hz, so a grasp late in the approach still trains.
- `entropy_coef` 0.001. At 0.002 the policy kept exploring after the peak and knocked the strap out.

## Bag spawn

Each reset draws a new position and a new heading, uniform and independent per env. The reference spot is
`(0, -0.03)` m. `bag_x_range` is about ±18 cm. `bag_y_range` is about 3 cm to 21 cm farther into the tray
than that spot. `bag_yaw_range` is a full turn about vertical, ±180 deg.

The center stays inside the opening by the body's diagonal (about 11 cm) plus a 5 mm gap, so a bag turned
to any heading still clears the walls. The reference spot is too close to the near wall for that, which is
why spawns sit farther in. The bag stays on the same face: no flip, no tilt. The recorded teacher still
aims at the strap pose from the reference spot.

## Grasp test

The strap is rigid, part of the same body as the bag (`bag_single_body.usda`). A grasp is the gripper
commanded closed and both jaws touching that strap. The contact point has to land in the handle zone in
the bag frame (`handle_zone_half_x` 4 cm, `handle_zone_half_y` 1.5 cm, above `handle_zone_min_z` 7.8 cm).
Debounced: on after 3 hits, off after 15 misses (0.5 s).

## Camera

One RealSense D415 sits 0.5 m above the center of the tray, looking straight down. It publishes RGB and
depth at the color field of view (69° by 42°), at 160×90. A dome light fills the scene and a distant
light shines from above.

The student finds the strap in the RGB picture. That gives the handle's camera-frame x and y. Its z is a
depth saved from the recorded pinch (about 0.463 m), not a sample of the live depth image: the camera
looks straight down and the strap height does not change when the bag slides or yaws, and the strap is
only a few pixels wide. The depth image is still an input, in metres / 2 so 1 is the 2 m far clip. Stereo
depth closer than 5 cm is stored as a miss. The student also gets the 7 joint angles and the base origin
(`base_08q`) in the same camera frame, so the base, the strap's image position, and that saved depth are
one reach from the arm to the handle. Those 11 numbers are a separate vector, joined to the image features
after the convolution. The privileged teacher is the value function. It sees the arm joint angles, the
jaw-tip position and orientation, and the bag position and orientation. It does not see the camera.

The camera policy is a separate run, `handle_grab_camera`. Its checkpoints go to
`logs/rl_games/handle_grab_camera/` and do not replace the earlier state-only saves in
`logs/rl_games/handle_grab/`. The last of those is `logs/rl_games/handle_grab/no_camera_checkpoint.pth`
(epoch 3025, reward 29.3). That file does not load into the camera network. Earlier camera-policy
saves do not load either: this student sees only the overhead picture, and its pose vector is the joint
angles, the base in the camera frame, and the saved strap depth. Start this run without `--checkpoint`.
The recorded arm motion is still the residual teacher, so a new policy begins by replaying the demo.

## Commands

Run these from the Isaac Lab repo root. `--viz kit` opens the Isaac Sim window.

### Train

```bash
# camera policy, from scratch. Leaves logs/rl_games/handle_grab/ untouched.
uv run --extra rl-games isaaclab train --rl_library rl_games --task=Isaac-HandleGrab-Direct --num_envs 128 --max_iterations 8000

# continue the newest camera-policy save (also restores the adaptive learning rate)
uv run --extra rl-games isaaclab train --rl_library rl_games \
  --task=Isaac-HandleGrab-Direct --num_envs 128 --checkpoint latest
```

Logs: `logs/rl_games/handle_grab_camera/`. `--checkpoint latest` looks only in that folder. The earlier
state-only run stays in `logs/rl_games/handle_grab/`. The terminal line is grasp rate, collision rate, and
time to grasp. `(num_envs * 128) % 512 == 0`, so `--num_envs` must be a multiple of 4. Checkpoints are
written every 25 epochs.

Physics is 60 Hz with 2 steps per policy step, so the policy stays at 30 Hz. The strap and each jaw are cooked as 8 convex pieces. A step is spent in that contact solve. 128 environments is the count that fits the overhead picture and the convolution on a 16 GB GPU.

```bash
uv run tensorboard --logdir logs/rl_games/handle_grab_camera
```

Flags and overrides, appended to the train command:

- `--checkpoint latest` continues from the newest periodic save in `logs/rl_games/handle_grab_camera/`. `--checkpoint best` uses `handle_grab_camera.pth`, the best score after epoch 100. A `.pth` path uses that file. The state-only checkpoint is `logs/rl_games/handle_grab/no_camera_checkpoint.pth` and does not load here.
- `--max_iterations 5000` sets the epoch cap (default 4000). On a resume it must be higher than the epoch already reached.
- `--num_envs 128` changes the parallel count. It has to be a multiple of 4.
- `--seed 1` fixes the seed. `--seed -1` draws one.
- `--viz kit` opens Isaac Sim during training. `--max_visible_envs 4` draws only 4 envs.
- `--video` records clips. `--video_length 200` and `--video_interval 2000` are the clip length and the gap, in steps.
- `'env.bag_x_range=[-0.18,0.18]'`, `'env.bag_y_range=[0.03,0.21]'`, and `'env.bag_yaw_range=[-3.1416,3.1416]'` are the spawn box in meters and the heading in radians. Set each to `[0,0]` to pin that part.
- `env.action_scale=0.35` is the residual in radians. `env.stagger_first_episodes=false` starts every env together and is what froze the learning rate before.
- `agent.params.config.learning_rate=2e-4` and `agent.params.config.kl_threshold=0.01` set the adaptive rate. `agent.params.config.lr_schedule=constant` turns that schedule off. `agent.params.config.entropy_coef=0.001` is the exploration weight. `0.002` kept knocking the strap out.
- `--track` logs to Weights and Biases. It needs `--wandb-entity`. `--wandb-project-name` and `--wandb-name` are optional.

Leave `--sigma` off. Exploration is tuned at log-std −1.5.

### Play a checkpoint

```bash
uv run --extra rl-games isaaclab play --rl_library rl_games \
  --task=Isaac-HandleGrab-Direct --num_envs 4 --checkpoint latest --viz kit
```

`--checkpoint latest` loads the newest camera-policy save under `logs/rl_games/handle_grab_camera/`. Pass a
`.pth` path instead of `latest` to play a specific file. To play the state-only policy, pass
`logs/rl_games/handle_grab/no_camera_checkpoint.pth` only with the code from before the camera change.

### Watch the teacher

A zero action replays `grasp_poses.json`. The training default stops at `grasp_closed` and holds. The bag
moves and turns on every reset. Turn staggering off so this one env plays the approach from home
instead of starting partway through. Add `'env.bag_x_range=[0,0]' 'env.bag_y_range=[0,0]' 'env.bag_yaw_range=[0,0]'`
to pin the bag at the reference spot and heading.

```bash
# approach, open around the strap, close, hold
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit --max_steps 900 env.stagger_first_episodes=false

# already closed, then the lift
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit --max_steps 900 env.stagger_first_episodes=false \
  env.demo_start_waypoint=grasp_closed env.demo_stop_waypoint=lift

# the rest of the recorded sequence: lift, point the jaw up, return home
uv run python scripts/environments/zero_agent.py --task Isaac-HandleGrab-Direct \
  --num_envs 1 --viz kit --max_steps 1200 env.stagger_first_episodes=false \
  env.demo_start_waypoint=grasp_closed env.demo_stop_waypoint=return_home
```

Waypoints, in order: `pregrasp`, `grasp`, `grasp_closed`, `lift`, `ee_up`, `return_home`.
`env.demo_start_waypoint` is the pose the arm holds first. `env.demo_stop_waypoint` is the last move it plays.

### Pose the arm

`grasp_pose_tool.py` drives one env from `official_handle_grab/scene/grasp_poses.json`. It pins the bag at the nominal spot so a saved
edit lands in the same place. Live mode holds the `"show"` waypoint and reloads the file when you save it.

```bash
# hold whatever "show" names (grasp_closed)
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit

# hold one waypoint: pregrasp, grasp, grasp_closed, lift, ee_up, or return_home
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py --viz kit --show grasp

# replay through the close
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py \
  --viz kit --mode run --stop_after grasp_closed

# replay the whole sequence and write official_handle_grab/scene/grasp_trajectory.json
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py \
  --viz kit --mode run --save_traj

# write the held waypoint into official_handle_grab/scene/grasp_goal.json and exit
uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/grasp_pose_tool.py \
  --viz kit --show grasp_closed --record
```
