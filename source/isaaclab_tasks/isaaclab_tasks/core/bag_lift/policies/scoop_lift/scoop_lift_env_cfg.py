# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoop-lift policy: the soft bag is gripped by its strap and scooped up by a Cartesian teacher.

The episode starts from the closed grasp on the strap (IK per bag). The teacher is the waypoint list in
scoop_trajectory.json (see BagLiftEnv._build_scoop_teacher). Zero action replays it. The policy learns a small
residual around it. The bag starts at the same spot every reset.
"""

from isaaclab.utils import configclass

from ...bag_lift_env_cfg import BagLiftEnvCfg


@configclass
class ScoopLiftEnvCfg(BagLiftEnvCfg):
    report_name = "ScoopLift"
    report_focus = "lift"

    # Closed grasp on the strap at reset, jaws held shut, Cartesian scoop teacher, lift = success.
    spawn_closed_grasp = True
    hold_gripper_closed = True
    success_is_lift = True
    scoop_teacher = True
    use_demo_residual = False
    stagger_first_episodes = False
    arm_joint_noise = 0.0
    # Let the whole path play out: episode_length_s is set from the teacher (path + hold_s from scoop_trajectory.json).
    terminate_on_success = False
    # Residual around the teacher: 0.05 rad ~ 3 deg per joint.
    action_scale = 0.05

    # Same spot every reset.
    rand_curriculum = False
    bag_xy_range = (0.0, 0.0)
    bag_yaw_range = (0.0, 0.0)
    bag_tilt_range = 0.0
    bag_random_flip = False

    # Starter reward: stay gripped, lift the bag, turn the jaw up, and a bonus for the 5 cm lift.
    rew_scale_reach = 0.0
    rew_scale_align = 0.0
    rew_scale_reached = 0.0
    rew_scale_close = 0.0
    rew_scale_track = 0.0
    rew_scale_grasp = 5.0
    rew_scale_squeeze = 1.0
    rew_scale_ee_up = 3.0
    rew_scale_hand_up = 3.0
    rew_scale_lift = 5.0
    rew_scale_return = 0.0
    rew_scale_collision = 0.0  # PhysX reports no contact forces on a soft bag body
    rew_scale_action_rate = -0.01
    rew_scale_residual = -0.05
    rew_scale_time = 0.0
    rew_scale_knocked_away = -20.0
    rew_scale_success = 50.0
