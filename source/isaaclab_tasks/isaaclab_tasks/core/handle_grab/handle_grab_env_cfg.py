# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CONFIG for the handle-grab task: find the strap and close on it. Numbers only, no logic.

The logic lives in handle_grab_env.py; scene objects (arm, bag, tray) live in
``official_handle_grab/scene/handle_grab_scene.py``.
Any value here can be overridden from the CLI with Hydra, e.g.  env.rew_scale_collision=-10.0

The teacher plays grasp_poses.json through grasp_closed and then holds. Every reset moves the bag to a new
spot inside the tray (bag_x_range, bag_y_range) and turns it about vertical (bag_yaw_range). The teacher
still aims at the strap pose it was recorded on.
Reward is the debounced grasp and contact on the lower 95% of the bag body. Lift and carry are not paid.
"""

import math
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import (
    AMBIENT_LIGHT_CFG,
    ARM_CFG,
    ARM_CONTACT_CFG,
    ARM_GRIPPER_JOINT,
    BAG_CFG,
    BAG_X_RANGE,
    BAG_Y_RANGE,
    BAG_YAW_RANGE,
    BASE_BODY_NAME,
    CAMERA_HEIGHT_PX,
    CAMERA_WIDTH,
    D415_CFG,
    D415_DEPTH_MIN_M,
    EE_BODY_NAME,
    EE_TIP_OFFSET,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    HANDLE_CFG,
    HANDLE_INFO_JSON,
    JAW_INSERT_AXIS,
    JAW_POINTS_JSON,
    MOVING_JAW_BODY_NAME,
    OVERHEAD_LIGHT_CFG,
    SANDBOX_CFG,
    USE_DEFORMABLE_HANDLE,
    USE_SAND_PARTICLES,
)

if USE_SAND_PARTICLES:
    _PHYSICS_CFG = PhysxCfg(gpu_max_particle_contacts=2**24)
elif USE_DEFORMABLE_HANDLE:
    _PHYSICS_CFG = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)  # deformable contacts need a larger pair buffer
else:
    _PHYSICS_CFG = PhysxCfg()


@configclass
class HandleGrabSceneCfg(InteractiveSceneCfg):
    sandbox = SANDBOX_CFG  # plain rigid tray
    arm = ARM_CFG  # SO-101, physics layer switched on
    bag = BAG_CFG  # bag merged into one rigid body
    contact: ContactSensorCfg = ARM_CONTACT_CFG  # every arm link vs the bag
    camera = D415_CFG  # RealSense D415, RGB + depth, 0.5 m above the tray center
    ambient = AMBIENT_LIGHT_CFG  # fill light
    overhead = OVERHEAD_LIGHT_CFG  # distant light from above

    def __post_init__(self):
        super().__post_init__()
        if USE_DEFORMABLE_HANDLE:
            self.handle = HANDLE_CFG  # bendable strap clamped to the bag body (handle_grab_scene.py)


@configclass
class EventCfg:
    # Bag grip and servo stiffness are re-drawn every reset. Set a range to a single value to freeze it.
    bag_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bag", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("arm", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.25),
            "damping_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class HandleGrabEnvCfg(DirectRLEnvCfg):
    # --- timing: physics 60 Hz, policy 30 Hz --------------------------------
    # The arm's joint speed limit is 0.5 rad/s, so a 60 Hz step moves the tip under a millimetre, inside
    # the contact skin. 120 Hz with 4 substeps did twice the solver work for the same 30 Hz policy.
    decimation = 2
    # Placeholder: with use_demo_residual the env sets episode_length_s = demo length + episode_extra_s.
    episode_length_s = 30.0
    episode_extra_s = 4.0  # [s] time after the teacher ends to settle into the return_home box
    # Every env resets at step 0, so without this they all time out on the same step and each PPO rollout
    # (horizon_length << episode length) holds one slice of the demo: a few epochs of approach, then a few of
    # grasp, then many of carry, which overwrite each other. True truncates the FIRST episode of each env at a
    # random length (the teacher still starts at home), which spreads the envs over the whole demo from then on.
    stagger_first_episodes = True

    # --- sizes (MUST match handle_grab_env.py) ----------------------------------
    action_space = 7  # 6 arm joints + gripper
    # Student image, channels last: overhead RGB + depth. Joint angles, the base in that camera frame, and the
    # saved strap depth are a separate 11-vector (see HandleGrabEnv._get_observations).
    observation_space = [CAMERA_HEIGHT_PX, CAMERA_WIDTH, 3 + 1]
    depth_fail_m = D415_DEPTH_MIN_M  # D415 stereo depth inside this distance is not a measurement
    # Privileged teacher (the critic): arm joint angles, jaw tip pose, bag pose. No camera.
    state_space = 7 + 3 + 4 + 3 + 4

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        physics=_PHYSICS_CFG,
    )
    # --num_envs overrides num_envs. Keep it a multiple of 4 (see agents/rl_games_direct_ppo_cfg.yaml).
    scene: HandleGrabSceneCfg = HandleGrabSceneCfg(num_envs=128, env_spacing=2.5)

    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="abs"),
    )
    # The picture is already in [0, 1]. A per-pixel bias would repaint the whole camera.
    observation_noise_model = None

    # --- arm / hand ------------------------------------------------------------
    base_body_name = BASE_BODY_NAME  # CAD base; its position in the camera frame is a student input
    ee_body_name = EE_BODY_NAME  # link holding the rigid jaw
    ee_tip_offset = EE_TIP_OFFSET  # rigid-jaw tip in that link's frame [m]
    jaw_insert_axis = JAW_INSERT_AXIS  # direction the rigid jaw points (link -Y)
    moving_jaw_body_name = MOVING_JAW_BODY_NAME
    gripper_joint_name = ARM_GRIPPER_JOINT
    gripper_open_pos = GRIPPER_OPEN_POS
    gripper_closed_pos = GRIPPER_CLOSED_POS
    # Training: joint target = teacher(t) + action_scale * action. Zero residual replays grasp_poses.json.
    # 0.35 rad (~20 deg) is enough to correct for the bag randomization below, not enough to throw the demo away.
    # The pose tool sets use_demo_residual False and widens action_scale so it can jump to any edited waypoint.
    use_demo_residual = True
    action_scale = 0.35
    # Extra range on revolute_4 (wrist pitch), in radians. 0 uses action_scale for that joint too.
    # The bag-lift policy needs this so the rigid jaw can turn toward world +Z. The other joints stay
    # on action_scale, so the arm is not free to fling.
    wrist_action_scale = 0.0
    # No carry, so the residual stays at action_scale instead of shrinking near return_home.
    residual_fade_frac = 0.0
    carry_action_scale = 0.07  # [rad] ~4 deg, under final_pose_tol so it cannot push the arm out of the box
    demo_speed = 1.0  # teacher playback: 1 = grasp_poses.json timing; >1 finishes the demo earlier (speed ceiling)
    # Last waypoint included in the teacher. The jaws close on the strap, then that pose is held.
    demo_stop_waypoint: str | None = "grasp_closed"
    # Pose the teacher starts from. None starts at the home pose and includes the first move. When set, the
    # teacher holds that pose, then plays the waypoints after it through demo_stop_waypoint.
    demo_start_waypoint: str | None = None
    # Arm joints written on reset. None keeps the USD home pose.
    reset_pose_waypoint: str | None = None
    # Place the closed gripper on this bag's strap at reset (bag frame, so it follows the bag pose). The arm
    # appears already shut: no approach and no closing motion. Off for the grasp-handle and full-lift tasks.
    spawn_closed_grasp = False
    # Keep the gripper joint at gripper_closed_pos. The policy's 6 arm joints still move. Off unless a policy
    # starts already shut and should not be able to open the jaw.
    hold_gripper_closed = False
    # Success is the bag lift_height up while the grip is still true, with no return-home pose. The full task
    # leaves this off and still requires return_home.
    success_is_lift = False
    spawn_press = 0.0  # [m] extra push of the closed jaws into the strap. 0 leaves them on the strap as solved.
    strap_half_width = 0.0065  # [m] half the strap width along bag +Y
    strap_apex = (0.0, 0.0, 0.11375)  # [m] top of the strap arch, bag frame
    # Teacher lifts the spawned grasp straight up by lift_height, jaw orientation unchanged. Curling
    # the jaw off this grasp puts the wrist where the arm cannot reach, and the hand leaves the strap.
    # Success is the bag up by lift_height while the grip is still true.
    ee_up_teacher = False
    # When set, the teacher moves from the spawned grasp by the joint change recorded between
    # approach_pose_waypoint and this waypoint (the "lift" pose in grasp_poses.json). None keeps the
    # straight Cartesian raise.
    lift_pose_waypoint: str | None = None
    # How far the teacher drives the end effector up in world z. 0 uses lift_height. The bag goal stays
    # lift_height; this climb is longer so a strap that only stretches cannot hide the motion.
    ee_climb = 0.0
    # Seconds the teacher takes to finish ee_climb. 0 spreads the climb over the whole episode.
    ee_climb_s = 0.0
    # Pay for the jaw axis rotating from its spawn direction toward world +Z, instead of only once it
    # is already above the horizontal. The full task leaves this off.
    ee_up_from_start = False
    ee_up_segments = 1  # knots between the grasp and the end of the lift path
    ee_up_speed = 0.40  # [rad/s] joint speed cap along that path (smoothstep peaks at 1.5x)
    ee_up_success = 0.985  # jaw axis · world +Z. 0.985 is about 10 deg off straight up.
    # End the episode when the geometric grasp is false. The grace is in policy steps from this reset's teacher
    # clock, so a spawned grasp can register before a false grasp ends the attempt.
    terminate_on_grasp_loss = False
    grasp_loss_grace_steps = 0
    # End the episode after x, y, and z of the bag velocity each stay under bag_still_speed for this long.
    # 0 disables it.
    bag_still_s = 0.0
    bag_still_speed = 0.001  # [m/s] 1 mm/s on an axis counts as stopped
    arm_joint_noise = 0.05  # [rad] start-pose jitter

    # --- handle geometry (bag frame) -------------------------------------------
    # The handle is a thin strap (13 mm wide along bag +Y, about 1 mm thick) bent into a U that sticks 38 mm out
    # of the bag face (bag +Z, so the apex is at z = 0.114). The rigid jaw is a wedge (9 mm at the tip, 25 mm
    # 5 cm up, then a bulky housing), so it cannot be threaded through the loop. The jaws pinch the strap instead.
    # Pinch target: strap apex, 12 mm below its lower edge, so the thin tip ends up beside the strap.
    pinch_target = (0.0, -0.012, 0.1135)  # [m] bag frame
    # Written by `grasp_pose_tool.py --record`. When it exists, pinch_target above is replaced by the recorded tip
    # position and the align reward matches the recorded jaw orientation (both in the bag frame).
    goal_file = str(Path(__file__).parents[1] / "official_handle_grab" / "scene" / "grasp_goal.json")
    reach_radius = 0.015  # [m] tip this close to the pinch target counts as reached
    carry_radius = 0.02  # [m] hand-rise reward fades out as the tip moves this far off the strap

    # Rigid-handle fallback only (USE_DEFORMABLE_HANDLE=False): a contact counts as "on the handle strap"
    # from its location in the bag frame (x along the strap, y its 13 mm width, z outward from the body).
    # The body's face is at z = 0.076 and the strap spans z 0.076..0.114.
    handle_zone_half_x = 0.040  # [m]
    handle_zone_half_y = 0.015  # [m]
    handle_zone_min_z = 0.078  # [m]
    # Bag-frame +Z: body bottom -0.076 .. opening +0.076, handle above that (handle_grab_scene.py). Collision
    # penalises only the lower collision_lower_frac of the body; the top rim and the handle are free.
    bag_body_z_min = -0.076  # [m]
    bag_body_z_max = 0.076  # [m]
    collision_lower_frac = 0.95  # lower this fraction of the body is a collision; top (1-frac) + handle is not

    # --- deformable handle (only when USE_DEFORMABLE_HANDLE in handle_grab_scene.py) -------------------------------
    # PhysX cannot report contacts on deformables, so the grasp is inferred from geometry (no forces):
    #   1. the gripper is commanded closed (target within grip_closed_margin of its closed limit)
    #   2. EITHER (preferred) >= min_strap_nodes strap nodes REST on the rigid jaw (within pinch_dist of its proxy
    #      points in assets/jaw_points.json, either face of the paddle) at depth < mouth_mid_frac + mouth_tol, i.e.
    #      midway or further back; OR >= min_strap_nodes nodes lie BETWEEN the jaws (within pinch_dist of both, on
    #      opposite sides) at depth mouth_mid_frac +- mouth_tol.
    #   Depth: 0 = housing end (mouth_base_y), 1 = jaw tip (mouth_tip_y), in the rigid-jaw link frame (-Y is the
    #   jaw's long axis).
    handle_info_file = HANDLE_INFO_JSON
    jaw_points_file = JAW_POINTS_JSON
    touch_dist = 0.006  # [m] a jaw 'touches' the strap within this (status display only)
    pinch_dist = 0.008  # [m] each jaw within this of a strap node, on opposite sides = node is between the jaws
    min_strap_nodes = 3  # strap nodes that must be between the jaws
    grip_closed_margin = 0.05  # [rad] gripper target this close to the closed limit counts as 'commanded closed'
    # Debounce: the geometric test flickers while the strap shifts in the jaws. grasped turns on after
    # grasp_on_steps consecutive raw hits and off after grasp_off_steps consecutive misses (policy steps).
    grasp_on_steps = 3
    grasp_off_steps = 15  # 0.5 s at 30 Hz
    mouth_base_y = -0.040  # [m] rigid-jaw frame y of the housing end of the mouth (depth 0)
    mouth_tip_y = -0.095  # [m] rigid-jaw frame y of the jaw tip (depth 1)
    mouth_mid_frac = 0.5  # target depth: midway into the mouth
    # depth may be off by this much (+-16 mm on the 55 mm mouth). The demo carries the bag with the strap at
    # depth 0.62-0.75, so 0.2 (up to 0.7) flipped grasped on and off from run to run on the same demo.
    mouth_tol = 0.3

    # --- final pose (the success condition) -----------------------------------------------------------------------
    # Success = gripper commanded closed, a jaw within success_strap_gap of the strap, the bag lift_height up
    # (it only stays up if something holds it), and joints 1-6 within final_pose_tol of return_home.
    final_pose_file = str(Path(__file__).parents[1] / "official_handle_grab" / "scene" / "grasp_poses.json")
    approach_pose_waypoint = "grasp_closed"  # joint-space target until the strap is held
    final_pose_waypoint = "return_home"  # waypoint in that file (joint angles inherit like in grasp_pose_tool.py)
    final_pose_tol = math.radians(10.0)  # [rad] every one of joints 1-6 must be this close
    return_scale = 0.5  # [rad] RMS joint error where the return reward has dropped to 24%
    track_scale = 0.25  # [rad] RMS tracking error where the demo-follow reward has dropped to 24%
    demo_hold_s = 0.3  # [s] pause after each waypoint, also the grasp_pose_tool.py --hold_s default
    demo_move_s = 2.0  # [s] fallback move duration when a waypoint has no speed_deg_per_s
    demo_close_s = 1.5  # [s] fallback duration of the grasp_closed move
    demo_min_move_s = 0.3  # [s] shortest move, same as grasp_pose_tool.MIN_MOVE_S

    # --- success ----------------------------------------------------------------
    terminate_on_success = False  # hold the closed grasp; this task does not end on the lift success test
    success_strap_gap = 0.010  # [m] closest jaw within this of a strap node counts as still on the strap
    lift_height = 0.05  # [m] bag must rise this far above its resting height while grasped
    bag_out_of_bounds = 0.35  # [m] bag pushed this far from the tray centre = episode over

    # --- collision penalty ---------------------------------------------------------
    # Per step: collision = tanh(F / collision_force_scale), F = contact force [N] on the LOWER
    # collision_lower_frac of the bag body. The top rim and the handle are not penalised.
    collision_force_scale = 5.0  # [N] the open-loop demo brushes the body at ~1 N; don't make that a cliff
    # The penalty weight ramps linearly 0 -> rew_scale_collision over this many env steps (1 PPO iteration =
    # horizon_length = 64 steps, so 16000 steps ~ 250 iterations). At full weight from step 0 the policy learned
    # to stay away from the bag instead of approaching it.
    collision_ramp_steps = 16000

    # --- bag reset randomization (offset from the nominal spawn, per episode) --------
    # xy is uniform over the tray opening, inset by the body diagonal so any heading stays inside the walls.
    # yaw is a full turn about world z. Flip and tilt stay fixed, so the bag stays lying on the same face.
    bag_x_range = BAG_X_RANGE  # [m]
    bag_y_range = BAG_Y_RANGE  # [m]
    bag_yaw_range = BAG_YAW_RANGE  # [rad] about world z
    bag_random_flip = False  # keep the handle on the same face
    bag_tilt_range = 0.0  # [rad]
    # Curriculum: the ranges above are multiplied by a scale that starts at rand_scale_init and steps up by
    # rand_scale_step each time a window of ~num_envs episodes reaches rand_curriculum_success final-pose rate.
    # rand_curriculum = False uses the full ranges from the start.
    rand_curriculum = False
    rand_scale_init = 0.0
    rand_scale_step = 0.1
    rand_curriculum_success = 0.8

    # Terminal line. "full" prints every stage. "grasp" prints grasp rate and the lower-body collision rate.
    # report_name is the tag in that line, so each policy folder can name its own runs.
    report_name = "HandleGrab"
    report_focus = "grasp"

    # --- reward weights (per-step terms are multiplied by step_dt; success is a one-off) ----------
    # reach / align / reached only pay before the grasp, so they do not pull the hand back down during the carry
    rew_scale_reach = 0.0  # rigid tip toward the pinch target on the strap
    rew_scale_align = 0.0  # rigid jaw matching the recorded goal orientation
    rew_scale_reached = 0.0  # rigid tip at the pinch target
    rew_scale_close = 0.0  # gripper closing while a jaw is at the strap (leads to the grasp)
    # All 7 joints toward the teacher pose this step. Off: it pays for NOT correcting for the bag pose, which is the
    # whole job of the residual (the demo alone reaches 38% at full randomization). The zero-initialised action head
    # and rew_scale_residual already keep the policy on the demo. Restore with env.rew_scale_track=2.0.
    rew_scale_track = 0.0
    rew_scale_grasp = 5.0  # geometric grasp (must beat hovering at the pinch without holding)
    rew_scale_squeeze = 0.0  # gripper closed fraction while grasped
    rew_scale_ee_up = 0.0  # rigid jaw pointing world +Z while grasped (carry the bag with the EE up)
    rew_scale_hand_up = 0.0  # hand rising 0..lift_height while grasped (dense: pays at the first cm)
    rew_scale_lift = 0.0  # bag rising 0..lift_height while grasped
    rew_scale_return = 0.0  # arm toward return_home while still holding the strap
    rew_scale_collision = -0.5  # contact on the lower 95% of the bag body (full weight after the ramp)
    rew_scale_action_rate = 0.0  # jerky motion
    rew_scale_residual = 0.0  # ||action||^2: stay on the teacher unless the bag draw requires a correction
    rew_scale_time = 0.0  # per second of wall time
    # One-off when the bag is pushed past bag_out_of_bounds (which ends the episode). Must outweigh the living cost
    # that ending early saves (|rew_scale_time| * episode length, ~62 at 31 s), else knocking the bag away pays.
    rew_scale_knocked_away = 0.0
    # One-off when the geometric grasp goes false (the episode ends on that step when terminate_on_grasp_loss).
    # Not scaled by step_dt. 0 leaves the full task and the grasp-handle policy unchanged.
    rew_scale_drop = 0.0
    # One-off per centimeter of upward bag motion, paid once, capped at lift_height. Not scaled by step_dt.
    # 0 keeps the dense rew_scale_lift term as the only lift reward.
    rew_per_cm = 0.0
    # One-off per centimeter the end effector rises in world z, paid once, while the grip holds and the bag
    # is still below lift_height. 0 leaves the full task and the grasp-handle policy unchanged.
    rew_per_arm_cm = 0.0
    # Centimeters of rigid-jaw height above the spawn, per second, while the grip holds. Scaled by step_dt.
    rew_scale_jaw_height = 0.0
    # Per centimeter the tip moves down this step. Not scaled by step_dt. 0 leaves the other tasks unchanged.
    rew_scale_jaw_down = 0.0
    # Per centimeter this episode's peak hand height finishes below the best episode so far. Paid once at the
    # end, not scaled by step_dt. The bar does not fall. 0 leaves the other tasks unchanged.
    rew_scale_hand_regress = 0.0
    # Per second the tip does not move in z, while the grip holds. Scaled by step_dt.
    rew_scale_jaw_stall = 0.0
    jaw_min_rise_cm_s = 0.5
    # One-off on first success, NOT scaled by step_dt. Must beat what staying grasped would still earn: the
    # holding-gated terms total ~28/s and at most ~episode_extra_s remain after the demo, so ~110.
    rew_scale_success = 0.0
