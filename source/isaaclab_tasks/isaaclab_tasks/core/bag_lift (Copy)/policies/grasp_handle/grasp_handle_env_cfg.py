# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-handle policy: the bag spawns at its nominal pose, and the reward is the grasp plus collision.

The grasp test is unchanged (BagLiftEnv: gripper commanded closed, strap held geometrically, debounced).
Collision is unchanged: contact force on the lower 95% of the bag body. The top rim and the handle are free.
The teacher is grasp_poses.json played through grasp_closed and then held. Lift and return are not in this task.
"""

from isaaclab.utils import configclass

from ...bag_lift_env_cfg import BagLiftEnvCfg


@configclass
class GraspHandleEnvCfg(BagLiftEnvCfg):
    report_name = "GraspHandle"
    report_focus = "grasp"
    # Teacher stops once the jaws have closed on the strap. The pose is held for the rest of the episode.
    demo_stop_waypoint = "grasp_closed"
    # No carry, so the residual stays at action_scale instead of shrinking near return_home.
    residual_fade_frac = 0.0
    terminate_on_success = False

    # Same spot every reset. rand_curriculum False would otherwise apply the full ranges; the ranges themselves are 0.
    rand_curriculum = False
    bag_xy_range = (0.0, 0.0)
    bag_yaw_range = (0.0, 0.0)
    bag_tilt_range = 0.0
    bag_random_flip = False

    # Lower 95% of the body. Top rim and handle are not collisions.
    collision_lower_frac = 0.95

    # Only the grasp (while the debounced grasp holds) and the ramped lower-body collision penalty.
    rew_scale_reach = 0.0
    rew_scale_align = 0.0
    rew_scale_reached = 0.0
    rew_scale_close = 0.0
    rew_scale_track = 0.0
    rew_scale_grasp = 5.0
    rew_scale_squeeze = 0.0
    rew_scale_ee_up = 0.0
    rew_scale_hand_up = 0.0
    rew_scale_lift = 0.0
    rew_scale_return = 0.0
    rew_scale_collision = -0.5
    rew_scale_action_rate = 0.0
    rew_scale_residual = 0.0
    rew_scale_time = 0.0
    rew_scale_knocked_away = 0.0
    rew_scale_success = 0.0
