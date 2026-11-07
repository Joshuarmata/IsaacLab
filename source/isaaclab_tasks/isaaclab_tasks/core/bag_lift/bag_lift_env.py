# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LOGIC for the bag-lift task (Direct workflow). Numbers live in bag_lift_env_cfg.py.

Every policy step Isaac Lab calls, in order:
    _pre_physics_step -> _apply_action (x decimation, physics) -> _get_dones -> _get_rewards
    -> _reset_idx (finished envs only) -> _get_observations
Every tensor has shape (num_envs, ...). No Python loops over envs.

TASK
    grasp : gripper commanded closed, and the strap held geometrically (no force threshold), debounced by
            cfg.grasp_on_steps / cfg.grasp_off_steps.
              deformable handle: strap nodes REST on the rigid jaw, or lie BETWEEN the two jaws
                                 (cfg.pinch_dist, mouth_mid_frac +- mouth_tol). PhysX cannot report
                                 contacts on deformables.
              rigid handle     : both jaws contacting the strap by contact-point location (handle_zone_*)
    lift  : while grasped, the bag is 5 cm above its resting height (a progress metric and a reward stage)
    final : SUCCESS = gripper commanded closed, a jaw within cfg.success_strap_gap of the strap, the bag still
            5 cm up, and arm joints 1-6 within cfg.final_pose_tol (10 deg) of the hand-posed final pose
            (grasp_poses.json, waypoint cfg.final_pose_waypoint = "return_home")
    The episode ends on first success (flat bonus) or at demo length + cfg.episode_extra_s.
    Contact on the lower 95% of the bag body is penalised. The top rim and the handle are free.

HOW TO EDIT
    * What counts as grasp / lift / collision: _update_task_state
    * Reward shaping: _get_rewards          * Policy observation: _get_observations
    * Joint targets from actions: _pre_physics_step     * Bag reset pose: _reset_idx
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from .bag_lift_env_cfg import BagLiftEnvCfg


class BagLiftEnv(DirectRLEnv):
    cfg: BagLiftEnvCfg

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    def __init__(self, cfg: BagLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.arm = self.scene["arm"]
        self.bag = self.scene["bag"]
        # Soft body (bag_sand_scene.USE_DEFORMABLE_BAG_BODY) or the original rigid one? A soft body has no root pose;
        # _bag_pose() fits one to its nodes. PhysX cannot report contacts on deformables: no contact sensor then.
        self._deformable_body = "bag" in self.scene.deformable_objects
        self._contact = None if self._deformable_body else self.scene["contact"]
        # Deformable strap (bag_sand_scene.USE_DEFORMABLE_HANDLE) or the original rigid one?
        self._deformable_handle = "handle" in self.scene.deformable_objects
        if self._deformable_handle:
            self.handle = self.scene["handle"]
        if self._deformable_body:
            self._init_deformable_body()

        ee_ids, _ = self.arm.find_bodies(self.cfg.ee_body_name)
        self._ee_idx = ee_ids[0]
        self._ee_offset = torch.tensor(self.cfg.ee_tip_offset, device=self.device).repeat(self.num_envs, 1)

        joint_ids, joint_names = self.arm.find_joints(".*")
        assert len(joint_ids) == self.cfg.action_space, (
            f"Arm has {len(joint_ids)} joints {joint_names}, but cfg.action_space = {self.cfg.action_space}."
        )
        grip_ids, _ = self.arm.find_joints(self.cfg.gripper_joint_name)
        self._grip_idx = grip_ids[0]
        self._shoulder_ids = self.arm.find_joints("revolute_[1-3]")[0]  # base, shoulder, elbow
        self._wrist_ids = self.arm.find_joints("revolute_[4-6]")[0]  # wrist pitch, yaw, roll: they turn the jaw
        self._joint_targets = self.arm.data.default_joint_pos.torch.clone()  # last commanded joint targets [rad]
        self._arm_joint_ids = [i for i in range(len(joint_ids)) if i != self._grip_idx]  # joints 1-6 (not the gripper)
        # approach = the hand-posed pinch (before the arm carries the bag home); success = return_home
        self._grasp_pose = self._load_waypoint(joint_names, self.cfg.approach_pose_waypoint)
        self._final_pose = self._load_waypoint(joint_names, self.cfg.final_pose_waypoint)
        self._reset_pose = (
            self._load_waypoint(joint_names, self.cfg.reset_pose_waypoint) if self.cfg.reset_pose_waypoint else None
        )
        self._demo_q = self._bake_demo(joint_names)  # (T, 7) teacher joint targets, one per policy step
        self._demo_len = int(self._demo_q.shape[0])
        self._teacher = self._demo_q[0].unsqueeze(0).expand(self.num_envs, -1).clone()
        # Closed-grasp spawn and the per-bag scoop teacher (cfg.scoop_*). Filled on reset when those modes are on:
        # _q_path holds the IK joint targets at the path knots, _path_s the seconds of each segment between knots.
        self._q_grasp = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._scoop_wps: list[dict] = []
        self._scoop_hold_s = 0.0
        if self.cfg.scoop_teacher:
            self._load_scoop_trajectory()
        n_seg = max(sum(int(w["knots"]) for w in self._scoop_wps), 1)
        self._q_path = torch.zeros(self.num_envs, n_seg + 1, self.cfg.action_space, device=self.device)
        self._path_s = torch.ones(self.num_envs, n_seg, device=self.device)
        self._mouth_local: torch.Tensor | None = None  # rigid jaw -> moving jaw, wrist frame
        self._face_local: torch.Tensor | None = None  # rigid-jaw face point, wrist frame
        self._moving_face_local: torch.Tensor | None = None  # opposing moving-jaw point, wrist frame
        # Roll of the closed mouth about the hole axis. Chosen once, the way MoveIt's compute_ik keeps the
        # solution nearest a comfortable seed. None until the first spawn picks it.
        self._pinch_roll: float | None = None
        self._grasp_nominal = self._grasp_pose.clone()
        self._grasp_nominal[self._grip_idx] = self.cfg.gripper_closed_pos
        self._spawn_announced = False
        self._jacobi_ee = self._ee_idx - (1 if self.arm.is_fixed_base else 0)
        self._arm_jac_cols = [i + self.arm.num_base_dofs for i in self._arm_joint_ids]
        self._insert_local = torch.tensor(self.cfg.jaw_insert_axis, device=self.device, dtype=torch.float32)
        # Teacher clock, one per env, zeroed on reset. It is separate from episode_length_buf so the first episodes
        # can be truncated at random lengths (cfg.stagger_first_episodes) while the demo still starts from home.
        self._demo_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.use_demo_residual and not self.cfg.scoop_teacher:
            # max_episode_length is derived from cfg.episode_length_s, so this sets the timeout
            demo_s = self._demo_len * self.step_dt / max(self.cfg.demo_speed, 1e-6)
            self.cfg.episode_length_s = demo_s + self.cfg.episode_extra_s
            print(
                f"[BagLift] episode length {self.cfg.episode_length_s:.1f} s = demo {demo_s:.1f} s + "
                f"{self.cfg.episode_extra_s:.1f} s ({self.max_episode_length} policy steps)",
                flush=True,
            )
        if self._deformable_handle:
            moving_ids, _ = self.arm.find_bodies(self.cfg.moving_jaw_body_name)
            self._moving_idx = moving_ids[0]
            with open(self.cfg.handle_info_file) as f:
                info = json.load(f)
            self._handle_rest = torch.tensor(info["rest_points"], device=self.device, dtype=torch.float32)  # bag frame
            with open(self.cfg.jaw_points_file) as f:
                jaws = json.load(f)
            self._jaw_pts_rigid = torch.tensor(jaws["rigid"], device=self.device, dtype=torch.float32)  # link frames
            self._jaw_pts_moving = torch.tensor(jaws["moving"], device=self.device, dtype=torch.float32)

        # Contact-sensor columns (one per arm link): which are the two jaws? (Rigid bag body only.)
        if self._contact is not None:
            names = self._contact.body_names
            self._rigid_jaw_col = names.index(self.cfg.ee_body_name)
            self._moving_jaw_col = names.index(self.cfg.moving_jaw_body_name)
            self._other_cols = torch.tensor(
                [i for i in range(len(names)) if i not in (self._rigid_jaw_col, self._moving_jaw_col)],
                device=self.device,
            )

        insert_axis = torch.tensor(self.cfg.jaw_insert_axis, device=self.device)
        self._insert_axis = (insert_axis / insert_axis.norm()).repeat(self.num_envs, 1)
        self._plate_normal = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )  # bag +Y = hole axis
        self._pinch_target = torch.tensor(self.cfg.pinch_target, device=self.device).repeat(self.num_envs, 1)
        self._goal_quat = (
            None  # recorded rigid-jaw orientation in the bag frame, (num_envs, 4); None = use 'point down'
        )
        if os.path.exists(self.cfg.goal_file):
            with open(self.cfg.goal_file) as f:
                goal = json.load(f)
            self._pinch_target = torch.tensor(goal["tip_pos_bag"], device=self.device).repeat(self.num_envs, 1)
            self._goal_quat = torch.tensor(goal["jaw_quat_bag"], device=self.device).repeat(self.num_envs, 1)
            # Recorded arm shape is the comfortable seed. The gripper in that file is open; the spawn shuts it.
            joints_deg = goal.get("joints_deg")
            if joints_deg:
                for i, name in enumerate(joint_names):
                    if name in joints_deg and i != self._grip_idx:
                        self._grasp_nominal[i] = math.radians(float(joints_deg[name]))
            print(
                f"[BagLift] goal pose loaded from {self.cfg.goal_file}: tip {goal['tip_pos_bag']} (bag frame)",
                flush=True,
            )
        else:
            print("[BagLift] no grasp_goal.json: using cfg.pinch_target and 'jaw points down' alignment", flush=True)

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        # Per-step task state, filled by _update_task_state (called from _get_dones, used by _get_rewards).
        zeros_f = torch.zeros(self.num_envs, device=self.device)
        zeros_b = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bag_rest_z = zeros_f.clone()  # lowest bag height seen this episode = where it settled
        self._dist = zeros_f.clone()  # rigid tip -> pinch target on the strap
        self._collision_force = zeros_f.clone()  # [N] penalised contact force on the bag
        self._lift_amt = zeros_f.clone()  # [m] bag height above its resting height
        self._hand_rise = zeros_f.clone()  # [m] rigid tip height above its resting 'through the loop' height
        self._other_force = zeros_f.clone()  # [N] bag contact force on the non-jaw links
        self._jaw_off_handle_force = zeros_f.clone()  # [N] jaw force on the bag body (not the handle strap)
        self._strap_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # strap nodes between jaws
        self._strap_depth = zeros_f.clone()  # 0 = at the housing end of the jaw mouth, 1 = at the jaw tips
        self._strap_midway = zeros_b.clone()  # strap is pinched midway OR resting on the rigid jaw
        self._rest_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # nodes on rigid jaw
        self._rest_depth = zeros_f.clone()
        self._pose_rms = zeros_f.clone()  # [rad] RMS error of joints 1-6 from the final pose
        self._pose_max = zeros_f.clone()  # [rad] worst single joint error from the final pose
        self._approach_rms = zeros_f.clone()  # [rad] RMS error of all joints from the approach (grasp_closed) pose
        self._track_rms = zeros_f.clone()  # [rad] RMS error of all joints from the teacher pose this step
        # per jaw: touching the strap? and (deformable handle) the distance [m] from the jaw to the nearest strap node
        self._rigid_on = zeros_b.clone()
        self._moving_on = zeros_b.clone()
        self._rigid_gap = zeros_f.clone() + float("inf")
        self._moving_gap = zeros_f.clone() + float("inf")
        self._reached = zeros_b.clone()  # rigid tip at the pinch target
        self._closed = zeros_f.clone()  # gripper closed fraction, 0 open .. 1 closed
        self._grasp_raw = zeros_b.clone()  # this step's geometric test, before debouncing
        # > 0: consecutive raw hits, < 0: consecutive raw misses
        self._grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasped = zeros_b.clone()
        self._knocked_away = zeros_b.clone()  # bag pushed past cfg.bag_out_of_bounds (ends the episode)
        self._success = zeros_b.clone()  # closed on the strap, bag 5 cm up, at the final pose (10 deg)
        self._first_success = zeros_b.clone()  # rising edge of _success this episode (pays the bonus once)
        self._time_to_success = zeros_f.clone()  # [s] wall time of first success; 0 if it never happened
        self._time_to_grasp = zeros_f.clone()  # [s] wall time of the first debounced grasp; 0 if it never happened
        # "ever this episode" flags -> the success rates
        self._ever_reached = zeros_b.clone()
        self._ever_grasped = zeros_b.clone()
        self._ever_lifted = zeros_b.clone()  # grasped with the bag 5 cm up (stage 1 of the task)
        self._ever_at_pose = zeros_b.clone()  # joints 1-6 within 10 deg of return_home at some point
        self._ever_success = zeros_b.clone()  # grasped and at the final pose at some point
        self._ever_collided = zeros_b.clone()
        self._jaw_z = zeros_f.clone()  # rigid-jaw long axis, world z component, -1..1
        self._ee_up = zeros_f.clone()  # rigid-jaw long axis toward world +Z, 0..1
        self._ee_up_max = zeros_f.clone()  # best ee-up this episode while grasped

        # Terminal progress line: sums over finished episodes, printed every ~num_envs of them.
        self._win_count = 0
        self._win_sums = torch.zeros(7, device=self.device)  # reach, grasp, lift, success, at-pose, collision, knocked
        self._win_tts_sum = 0.0  # time-to-success [s], successes only
        self._win_tts_n = 0
        self._win_grasp_tts_sum = 0.0  # time-to-grasp [s], episodes that grasped
        self._win_grasp_tts_n = 0
        self._win_held_sum = 0.0  # episodes that ended still grasped
        self._win_ee_sum = 0.0  # sum of per-episode peak ee-up while grasped
        # multiplies the bag spawn ranges (cfg.rand_curriculum); 1 = full randomization
        self._rand_scale = self.cfg.rand_scale_init if self.cfg.rand_curriculum else 1.0

        if self.sim.has_gui:
            self.set_debug_vis(True)

    def _load_waypoint(self, joint_names: list[str], waypoint: str) -> torch.Tensor:
        """Joint angles [rad] of ``waypoint``, shape (num_joints,).

        Same rules as grasp_pose_tool.py: waypoints list only the joints they change and inherit the rest from the
        previous waypoint in ``"sequence"`` (the first inherits the home pose). Targets are clamped to the joint
        limits. Incremental actions can reach any in-limit pose, so there is no action_scale reachability check.
        """
        with open(self.cfg.final_pose_file) as f:
            poses = json.load(f)
        current = self.arm.data.default_joint_pos.torch[0].clone()
        target = None
        for name in poses.get("sequence", ["pregrasp", "grasp", "grasp_closed", "lift"]):
            for joint, deg in poses.get(name, {}).items():
                current[joint_names.index(joint)] = math.radians(float(deg))
            if name == waypoint:
                target = current.clone()
                break
        if target is None:
            raise ValueError(
                f"[BagLift] waypoint '{waypoint}' is not in the \"sequence\" of {self.cfg.final_pose_file}"
            )
        limits = self.arm.data.soft_joint_pos_limits.torch[0]
        target = target.clamp(limits[:, 0], limits[:, 1])
        print(
            f"[BagLift] pose '{waypoint}' (deg): "
            + ", ".join(f"{n[-1]}:{math.degrees(float(v)):.1f}" for n, v in zip(joint_names, target)),
            flush=True,
        )
        return target

    def _bake_demo(self, joint_names: list[str]) -> torch.Tensor:
        """Joint-space teacher trajectory (T, 7) from grasp_poses.json, same interpolation as grasp_pose_tool.py.

        Each waypoint is a smoothstep move at ``speed_deg_per_s``, then a hold. After the last waypoint the last
        pose is held, so indexing past T-1 is safe. A zero residual then replays the successful open-loop demo.
        """
        cfg = self.cfg
        dt = float(self.step_dt)
        with open(cfg.final_pose_file) as f:
            poses = json.load(f)
        home = self.arm.data.default_joint_pos.torch[0].clone()
        current = home.clone()
        full = list(poses.get("sequence", ["pregrasp", "grasp", "grasp_closed", "lift"]))
        all_poses: list[torch.Tensor] = []
        for name in full:
            for joint, deg in poses.get(name, {}).items():
                current[joint_names.index(joint)] = math.radians(float(deg))
            all_poses.append(current.clone())
        limits = self.arm.data.soft_joint_pos_limits.torch[0]
        all_poses = [w.clamp(limits[:, 0], limits[:, 1]) for w in all_poses]
        start = cfg.demo_start_waypoint
        stop = cfg.demo_stop_waypoint
        if start is not None and start not in full:
            raise ValueError(
                f"[BagLift] demo_start_waypoint '{start}' is not in the sequence of {cfg.final_pose_file}: {full}"
            )
        if stop is not None and stop not in full:
            raise ValueError(
                f"[BagLift] demo_stop_waypoint '{stop}' is not in the sequence of {cfg.final_pose_file}: {full}"
            )
        start_i = full.index(start) if start is not None else None
        stop_i = full.index(stop) if stop is not None else len(full) - 1
        if start_i is not None and start_i > stop_i:
            raise ValueError(f"[BagLift] demo starts at '{start}' which is after the stop waypoint '{stop}'")
        if start_i is None:
            sequence = full[: stop_i + 1]
            waypoints = all_poses[: stop_i + 1]
            prev = home
            rows: list[torch.Tensor] = []
        else:
            # Hold the start pose so a spawned grasp can register, then play the waypoints after it.
            sequence = full[start_i + 1 : stop_i + 1]
            waypoints = all_poses[start_i + 1 : stop_i + 1]
            prev = all_poses[start_i]
            rows = [prev] * max(1, int(round(cfg.demo_hold_s / dt)))

        speeds = poses.get("speed_deg_per_s", {})
        for name, goal in zip(sequence, waypoints):
            delta_deg = torch.rad2deg(goal - prev)
            spec = speeds.get(name, {})
            if not isinstance(spec, dict):
                spec = {"default": spec}
            fallback = cfg.demo_close_s if name == "grasp_closed" else cfg.demo_move_s
            per_joint = torch.tensor(
                [float(spec.get(j, spec.get("default", 0.0))) for j in joint_names],
                device=self.device,
                dtype=home.dtype,
            )
            secs = torch.where(
                per_joint > 0.0, delta_deg.abs() / per_joint.clamp(min=1e-6), torch.zeros_like(per_joint)
            )
            stage_s = max(cfg.demo_min_move_s, float(secs.max())) if float(per_joint.max()) > 0.0 else fallback
            secs = torch.where(per_joint > 0.0, secs, torch.full_like(secs, stage_s))
            steps = max(1, int(round(stage_s / dt)))
            durations = secs.clamp(min=1e-6)
            for k in range(1, steps + 1):
                s = (k * dt / durations).clamp(0.0, 1.0)
                s = s * s * (3.0 - 2.0 * s)
                rows.append(prev + s * (goal - prev))
            rows.extend([goal] * max(1, int(round(cfg.demo_hold_s / dt))))
            prev = goal
        demo = torch.stack(rows, dim=0)
        start_txt = f" from '{start}'" if start is not None else ""
        stop_txt = f" through '{stop}'" if stop is not None else ""
        print(
            f"[BagLift] demo from {cfg.final_pose_file}{start_txt}{stop_txt}: {len(sequence)} moves, "
            f"{demo.shape[0]} steps ({demo.shape[0] * dt:.1f} s). Residual {'ON' if cfg.use_demo_residual else 'off'}.",
            flush=True,
        )
        return demo

    def _setup_scene(self) -> None:
        # The bag rests on TrayFloor at z = 0. Strip Arm vs TrayFloor collisions so the jaw can pass through
        # the visual floor by EE_FLOOR_CLEARANCE (TrayEeLimit then stops it). pxr needs Kit up, so import here.
        from pxr import Usd, UsdPhysics

        stage = self.scene.stage
        for env_path in self.scene.env_prim_paths:
            floor = stage.GetPrimAtPath(f"{env_path}/Sandbox/TrayFloor")
            arm = stage.GetPrimAtPath(f"{env_path}/Arm")
            if not floor.IsValid() or not arm.IsValid():
                continue
            floor_rel = UsdPhysics.FilteredPairsAPI.Apply(floor).CreateFilteredPairsRel()
            for prim in Usd.PrimRange(arm, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    floor_rel.AddTarget(prim.GetPath())
                    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(floor.GetPath())
        if "handle" in self.scene.deformable_objects:
            self._author_handle_attachment()

    def _author_handle_attachment(self) -> None:
        """Clamp the strap's two feet to the bag body, in every env (the cloner cannot remap these relationships).

        Authors two USD prims per env, using the OmniPhysics deformable schemas PhysX reads when the sim starts:
          * HandleAttachment: OmniPhysicsVtxXformAttachment. Ties the strap vertices in ``anchor_vertices`` (the
            bottom ~8 mm of each leg, which the ribbon extends 4 mm into the body) to the rigid bag's frame, as a hard
            two-way constraint (stiffness = inf).
          * HandleCollisionFilter: OmniPhysicsElementCollisionFilter. The triangles around those clamped vertices
            skip collision with the body, otherwise they fight their own attachment.
        The vertex order of the strap's simulation mesh is the vertex order of assets/bag_handle_ribbon.usda.
        """
        from pxr import Sdf, Usd, UsdPhysics, Vt

        with open(self.cfg.handle_info_file) as f:
            info = json.load(f)
        anchors = [int(i) for i in info["anchor_vertices"]]
        local = np.asarray(info["rest_points"], dtype=np.float32)[anchors]
        triangles = [int(i) for i in info["anchor_triangles"]]

        stage = self.scene.stage
        soft = "bag" in self.scene.deformable_objects  # self._deformable_body is not set yet at scene setup
        tet_ids = tet_coords = None
        if soft:
            tet_ids, tet_coords = self._anchor_tet_coordinates(local)
        for env_path in self.scene.env_prim_paths:
            bag = stage.GetPrimAtPath(f"{env_path}/Bag")
            sim_mesh = stage.GetPrimAtPath(f"{env_path}/Handle/sim_mesh")
            if not bag.IsValid() or not sim_mesh.IsValid():
                raise RuntimeError(
                    f"[BagLift] cannot clamp the deformable handle in {env_path}: missing {env_path}/Bag or "
                    f"{env_path}/Handle/sim_mesh. Check the Handle spawn (bag_sand_scene.HANDLE_CFG)."
                )
            if soft:
                # soft body: each strap vertex is pinned to a point inside one tetrahedron of the body's sim mesh
                body_sim = stage.GetPrimAtPath(f"{env_path}/Bag/sim_mesh")
                if not body_sim.IsValid():
                    raise RuntimeError(f"[BagLift] missing {env_path}/Bag/sim_mesh (soft bag body did not spawn)")
                attachment = stage.DefinePrim(f"{env_path}/HandleAttachment", "OmniPhysicsVtxTetAttachment")
                attachment.CreateRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
                attachment.CreateRelationship("omniphysics:src1").SetTargets([body_sim.GetPath()])
                attachment.CreateAttribute("omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray).Set(
                    Vt.IntArray(anchors)
                )
                attachment.CreateAttribute("omniphysics:tetIndicesSrc1", Sdf.ValueTypeNames.IntArray).Set(
                    Vt.IntArray(tet_ids)
                )
                attachment.CreateAttribute("omniphysics:tetCoordsSrc1", Sdf.ValueTypeNames.Vector3fArray).Set(
                    Vt.Vec3fArray.FromNumpy(tet_coords)
                )
                attachment.CreateAttribute("omniphysics:stiffness", Sdf.ValueTypeNames.Float).Set(float("inf"))
            else:
                attachment = stage.DefinePrim(f"{env_path}/HandleAttachment", "OmniPhysicsVtxXformAttachment")
                attachment.CreateRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
                attachment.CreateRelationship("omniphysics:src1").SetTargets([bag.GetPath()])
                attachment.CreateAttribute("omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray).Set(
                    Vt.IntArray(anchors)
                )
                attachment.CreateAttribute("omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray).Set(
                    Vt.Vec3fArray.FromNumpy(local)
                )
                attachment.CreateAttribute("omniphysics:stiffness", Sdf.ValueTypeNames.Float).Set(float("inf"))

            colliders = [
                p for p in Usd.PrimRange(bag, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
            ]
            for k, collider in enumerate(colliders):
                flt = stage.DefinePrim(f"{env_path}/HandleCollisionFilter_{k}", "OmniPhysicsElementCollisionFilter")
                flt.CreateRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
                flt.CreateRelationship("omniphysics:src1").SetTargets([collider.GetPath()])
                # group 0: the strap triangles at the feet; src1 (the rigid body) has no groups = filtered as a whole
                flt.CreateAttribute("omniphysics:groupElemCounts0", Sdf.ValueTypeNames.UIntArray).Set(
                    Vt.UIntArray([len(triangles)])
                )
                flt.CreateAttribute("omniphysics:groupElemIndices0", Sdf.ValueTypeNames.UIntArray).Set(
                    Vt.UIntArray(triangles)
                )

    def _anchor_tet_coordinates(self, points: np.ndarray) -> tuple[list[int], np.ndarray]:
        """Tetrahedron index and barycentric (U, V, W) of each strap anchor point inside the soft bag body.

        ``points`` are in the bag frame, which is also the frame of the body's simulation mesh. PhysX's own tet
        finder is used when it is available, because it writes the exact coordinates the attachment schema expects
        (the asset validator authors attachments the same way). Otherwise the barycentric weights of the first three
        tet vertices are solved here. A point outside every tet is moved to the nearest one and reported.
        """
        from pxr import Usd

        stage = self.scene.stage
        mesh = stage.GetPrimAtPath(f"{self.scene.env_prim_paths[0]}/Bag/sim_mesh")
        if not mesh.IsValid():
            raise RuntimeError(f"[BagLift] missing {self.scene.env_prim_paths[0]}/Bag/sim_mesh (soft bag body)")
        verts = np.asarray(mesh.GetAttribute("points").Get(Usd.TimeCode.Default()), dtype=np.float64)
        tets = np.asarray(mesh.GetAttribute("tetVertexIndices").Get(Usd.TimeCode.Default()), dtype=np.int64)
        if verts.size == 0 or tets.size == 0:
            raise RuntimeError(
                "[BagLift] the soft bag body has no tetrahedral mesh. Install the tetrahedralization extra "
                "(uv sync --inexact --extra tetrahedralization) and rerun."
            )
        points = np.asarray(points, dtype=np.float64)

        try:
            from omni.physx.scripts.ifaces import acquire_physx_attachment_private_interface

            iface = acquire_physx_attachment_private_interface()
            finder = iface.create_tet_finder([tuple(map(float, v)) for v in verts], [int(i) for i in tets.reshape(-1)])
            out = iface.points_to_tetmesh_local(finder, [tuple(map(float, p)) for p in points])
            iface.release_tet_finder(finder)
            ids = [int(i) for i in out["tet_ids"]]
            coords = np.asarray([list(b)[:3] for b in out["barycentric_coords"]], dtype=np.float32)
            if len(ids) == len(points) and all(i >= 0 for i in ids):
                print(f"[BagLift] strap clamp: {len(ids)} anchors mapped into the bag tet mesh (PhysX tet finder)")
                return ids, coords
            print(f"[BagLift] PhysX tet finder missed {sum(i < 0 for i in ids)} anchors; solving them in Python")
        except Exception as err:  # noqa: BLE001 - optional private interface
            print(f"[BagLift] PhysX tet finder unavailable ({type(err).__name__}); solving the anchors in Python")

        p0, p1, p2, p3 = (verts[tets[:, k]] for k in range(4))
        basis = np.stack((p0 - p3, p1 - p3, p2 - p3), axis=-1)  # (T, 3, 3): weights of p0, p1, p2
        inv = np.linalg.inv(basis)
        ids, coords, worst = [], [], 0.0
        for p in points:
            w012 = np.einsum("tij,tj->ti", inv, p - p3)
            w = np.concatenate((w012, 1.0 - w012.sum(axis=1, keepdims=True)), axis=1)
            t = int(np.argmax(w.min(axis=1)))  # the tet that contains p, or the nearest one
            worst = max(worst, float(-w[t].min()))
            wc = np.clip(w[t], 0.0, None)
            wc /= wc.sum()
            ids.append(t)
            coords.append(wc[:3])
        print(
            f"[BagLift] strap clamp: {len(ids)} anchors mapped into the bag tet mesh (Python, "
            f"worst overshoot outside a tet {worst * 100:.1f}%)"
        )
        return ids, np.asarray(coords, dtype=np.float32)

    # -------------------------------------------------------------------------
    # GEOMETRY HELPERS (world frame; quaternions are (x, y, z, w))
    # -------------------------------------------------------------------------
    def _init_deformable_body(self) -> None:
        """Rest shape of the soft bag body in the bag frame, read once from the simulation mesh at spawn.

        The simulation mesh is tetrahedralized at spawn, so its node order is not known in advance. At construction
        the nodes still sit at the spawn pose (cfg init_state), so undoing that pose gives the rest nodes in the bag
        frame. They are what _bag_pose() fits against and what a reset writes back (rotated and translated).
        """
        init = self.cfg.scene.bag.init_state
        self._body_init_pos = torch.tensor(init.pos, device=self.device, dtype=torch.float32)
        self._body_init_quat = torch.tensor(init.rot, device=self.device, dtype=torch.float32)
        nodes = self.bag.data.nodal_pos_w.torch[0]  # (V, 3) world, env 0
        local = quat_apply_inverse(
            self._body_init_quat.expand(nodes.shape[0], 4), nodes - self.scene.env_origins[0] - self._body_init_pos
        )
        self._body_rest = local.clone()  # (V, 3) bag frame
        self._body_rest_mean = local.mean(dim=0)
        self._body_rest_c = local - self._body_rest_mean  # centred, for the Kabsch fit
        lo, hi = local.amin(dim=0), local.amax(dim=0)
        print(
            f"[BagLift] soft bag body: {local.shape[0]} simulation nodes, rest bbox (bag frame) "
            f"min {[round(float(v), 3) for v in lo]} max {[round(float(v), 3) for v in hi]} "
            f"(expected about [-0.083, -0.045, -0.076] .. [0.083, 0.045, 0.076])",
            flush=True,
        )

    def _bag_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bag frame pose in the world: ``(pos (n, 3), quat (n, 4) as x, y, z, w)``.

        A rigid bag has a root pose. A soft body has none, so the rotation and translation that best map the rest
        nodes onto the current nodes (least squares, Kabsch) are used. While the bag is only slightly deformed that
        is the pose of the bag; as it bends, it follows the average motion of the whole body.
        """
        if not self._deformable_body:
            return self.bag.data.root_pos_w.torch, self.bag.data.root_quat_w.torch
        nodes = self.bag.data.nodal_pos_w.torch  # (n, V, 3)
        mean = nodes.mean(dim=1)
        cov = torch.einsum("nvi,vj->nij", nodes - mean.unsqueeze(1), self._body_rest_c)  # sum cur * rest^T
        u, _, vh = torch.linalg.svd(cov)
        det = torch.linalg.det(u @ vh)
        flip = torch.ones(cov.shape[0], 3, device=self.device)
        flip[:, 2] = det.sign()
        rot = (u * flip.unsqueeze(1)) @ vh  # (n, 3, 3), proper rotation
        pos = mean - torch.einsum("nij,j->ni", rot, self._body_rest_mean)
        return pos, quat_from_matrix(rot)

    def _target_pos_w(self) -> torch.Tensor:
        """Where the rigid tip should go: on the strap's outermost point, just below its lower edge."""
        bag_pos, bag_quat = self._bag_pose()
        return bag_pos + quat_apply(bag_quat, self._pinch_target)

    def _ee_pos_w(self) -> torch.Tensor:
        """Rigid-jaw tip."""
        body_pos = self.arm.data.body_pos_w.torch[:, self._ee_idx]
        body_quat = self.arm.data.body_quat_w.torch[:, self._ee_idx]
        return body_pos + quat_apply(body_quat, self._ee_offset)

    def _hole_axis_w(self) -> torch.Tensor:
        """Unit axis perpendicular to the strap loop (bag +Y; world +Z once the bag lies flat)."""
        axis = quat_apply(self._bag_pose()[1], self._plate_normal)
        return axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _insert_alignment(self) -> torch.Tensor:
        """1 when the rigid jaw has the recorded goal orientation (or points straight down when there is no goal)."""
        if self._goal_quat is not None:
            goal_w = quat_mul(self._bag_pose()[1], self._goal_quat)
            err = quat_error_magnitude(self.arm.data.body_quat_w.torch[:, self._ee_idx], goal_w)  # [rad]
            return 1.0 - torch.tanh(err / 0.5)
        jaw_fwd = quat_apply(self.arm.data.body_quat_w.torch[:, self._ee_idx], self._insert_axis)
        return (0.5 * (1.0 - (jaw_fwd * self._hole_axis_w()).sum(dim=-1))).clamp(0.0, 1.0)

    def _gripper_closed_frac(self) -> torch.Tensor:
        """0 = gripper fully open, 1 = fully closed."""
        g = self.arm.data.joint_pos.torch[:, self._grip_idx]
        span = self.cfg.gripper_open_pos - self.cfg.gripper_closed_pos
        return ((self.cfg.gripper_open_pos - g) / span).clamp(0.0, 1.0)

    def _on_handle(self, link_col: int) -> torch.Tensor:
        """True when that link's contact point on the bag is on the handle strap (not the blue body)."""
        cfg = self.cfg
        # average contact point of this link with the bag, world frame, NaN when not touching
        pos_w = self._contact.data.contact_pos_w.torch[:, link_col, 0]
        rel = quat_apply_inverse(self.bag.data.root_quat_w.torch, pos_w - self.bag.data.root_pos_w.torch)
        return (
            (rel[:, 0].abs() < cfg.handle_zone_half_x)
            & (rel[:, 1].abs() < cfg.handle_zone_half_y)
            & (rel[:, 2] > cfg.handle_zone_min_z)
        )  # NaN (no contact) compares False

    def _on_lower_body(self) -> torch.Tensor:
        """(num_envs, num_links) True when that link is touching the lower collision_lower_frac of the bag body.

        Bag-frame +Z runs from the bottom (bag_body_z_min) to the opening (bag_body_z_max); the handle sits above
        that. Contacts on the top rim or the handle (z >= the cut) are not collisions. NaN (no contact) is False.
        """
        pos_w = self._contact.data.contact_pos_w.torch[:, :, 0]  # (num_envs, num_links, 3)
        n, n_links = pos_w.shape[:2]
        bag_p = self.bag.data.root_pos_w.torch.unsqueeze(1).expand(n, n_links, 3).reshape(-1, 3)
        bag_q = self.bag.data.root_quat_w.torch.unsqueeze(1).expand(n, n_links, 4).reshape(-1, 4)
        rel_z = quat_apply_inverse(bag_q, pos_w.reshape(-1, 3) - bag_p).reshape(n, n_links, 3)[..., 2]
        z_cut = self.cfg.bag_body_z_min + self.cfg.collision_lower_frac * (
            self.cfg.bag_body_z_max - self.cfg.bag_body_z_min
        )
        return rel_z < z_cut

    def _jaw_points_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World positions (num_envs, P, 3) of the proxy points on the rigid jaw and on the moving jaw."""
        body_pos = self.arm.data.body_pos_w.torch
        body_quat = self.arm.data.body_quat_w.torch
        out = []
        for idx, pts in ((self._ee_idx, self._jaw_pts_rigid), (self._moving_idx, self._jaw_pts_moving)):
            n, p = body_pos.shape[0], pts.shape[0]
            quat = body_quat[:, idx].unsqueeze(1).expand(n, p, 4).reshape(-1, 4)
            local = quat_apply(quat, pts.expand(n, p, 3).reshape(-1, 3)).reshape(n, p, 3)
            out.append(body_pos[:, idx].unsqueeze(1) + local)
        return out[0], out[1]

    def _strap_in_jaws(self) -> None:
        """Where is the strap relative to the two jaws? Fills the gaps, the 'between the jaws' count and its depth.

        A strap node is BETWEEN the jaws when the nearest rigid-jaw point and the nearest moving-jaw point are both
        within cfg.pinch_dist and lie on opposite sides of it (the two vectors node -> jaw point point away from each
        other). Depth = where those nodes sit along the rigid jaw's long axis: 0 at the housing end of the mouth
        (cfg.mouth_base_y), 1 at the tip (cfg.mouth_tip_y), so 0.5 is midway into the mouth.
        """
        cfg = self.cfg
        nodes = self.handle.data.nodal_pos_w.torch  # (num_envs, V, 3)
        rigid_w, moving_w = self._jaw_points_w()
        d_r = torch.cdist(nodes, rigid_w, compute_mode="donot_use_mm_for_euclid_dist")  # (num_envs, V, P)
        d_m = torch.cdist(nodes, moving_w, compute_mode="donot_use_mm_for_euclid_dist")
        near_r, idx_r = d_r.min(dim=-1)  # (num_envs, V)
        near_m, idx_m = d_m.min(dim=-1)
        self._rigid_gap = near_r.amin(dim=-1)
        self._moving_gap = near_m.amin(dim=-1)

        to_r = torch.gather(rigid_w, 1, idx_r.unsqueeze(-1).expand(-1, -1, 3)) - nodes
        to_m = torch.gather(moving_w, 1, idx_m.unsqueeze(-1).expand(-1, -1, 3)) - nodes
        between = (near_r < cfg.pinch_dist) & (near_m < cfg.pinch_dist) & ((to_r * to_m).sum(dim=-1) < 0.0)

        n, v = nodes.shape[:2]
        body_pos = self.arm.data.body_pos_w.torch[:, self._ee_idx]
        quat = self.arm.data.body_quat_w.torch[:, self._ee_idx].unsqueeze(1).expand(n, v, 4).reshape(-1, 4)
        in_jaw = quat_apply_inverse(quat, (nodes - body_pos.unsqueeze(1)).reshape(-1, 3)).reshape(n, v, 3)
        depth = (in_jaw[..., 1] - cfg.mouth_base_y) / (cfg.mouth_tip_y - cfg.mouth_base_y)  # 0 housing .. 1 tip
        count = between.sum(dim=-1)
        self._strap_count = count
        self._strap_depth = (depth * between).sum(dim=-1) / count.clamp(min=1)

        # RESTING on the rigid jaw: nodes within pinch_dist of it, either face. The paddle is thin and the
        # strap wraps it as the wrist turns, so a half-space test ("mouth side") drops a strap that is still
        # on the jaw. The proxy points only cover the paddle, so this does not count the rest of the arm.
        resting = near_r < cfg.pinch_dist
        rest_count = resting.sum(dim=-1)
        self._rest_count = rest_count
        self._rest_depth = (depth * resting).sum(dim=-1) / rest_count.clamp(min=1)

    def _link_forces(self) -> torch.Tensor:
        """(num_envs, num_links) contact force [N] each arm link exerts on the bag."""
        return torch.linalg.norm(self._contact.data.normal_force_matrix_w.torch, dim=-1).sum(dim=-1)

    # -------------------------------------------------------------------------
    # TASK STATE: grasp, lift, collision. The single place these are defined.
    # -------------------------------------------------------------------------
    def _update_task_state(self) -> None:
        cfg = self.cfg
        dist = torch.norm(self._ee_pos_w() - self._target_pos_w(), dim=-1)
        reached = dist < cfg.reach_radius

        closed_cmd = self._joint_targets[:, self._grip_idx] < cfg.gripper_closed_pos + cfg.grip_closed_margin
        if self._deformable_handle:
            # The contact sensor cannot see the deformable strap, so grasp is geometric (no force).
            self._strap_in_jaws()
            rigid_on = self._rigid_gap < cfg.touch_dist
            moving_on = self._moving_gap < cfg.touch_dist
            pinched = (self._strap_count >= cfg.min_strap_nodes) & (
                (self._strap_depth - cfg.mouth_mid_frac).abs() < cfg.mouth_tol
            )  # pinched between both jaws, about midway into the mouth
            resting = (self._rest_count >= cfg.min_strap_nodes) & (
                self._rest_depth < cfg.mouth_mid_frac + cfg.mouth_tol
            )  # resting on the rigid jaw, midway or further back (preferred)
            self._strap_midway = pinched | resting
            raw = closed_cmd & self._strap_midway
            near_strap = torch.minimum(self._rigid_gap, self._moving_gap) < cfg.success_strap_gap
        else:
            rigid_on = self._on_handle(self._rigid_jaw_col)
            moving_on = self._on_handle(self._moving_jaw_col)
            raw = closed_cmd & rigid_on & moving_on
            near_strap = raw
        self._rigid_on = rigid_on
        self._moving_on = moving_on
        # GRASP, debounced: on after grasp_on_steps raw hits in a row, off after grasp_off_steps misses in a row.
        self._grasp_raw = raw
        self._grasp_streak = torch.where(raw, self._grasp_streak.clamp(min=0) + 1, self._grasp_streak.clamp(max=0) - 1)
        self._grasped = torch.where(
            self._grasp_streak >= cfg.grasp_on_steps,
            True,
            torch.where(self._grasp_streak <= -cfg.grasp_off_steps, False, self._grasped),
        )
        # COLLISION: force on the lower collision_lower_frac of the bag body. The top rim and the handle are free
        # (the jaws have to sit there). Deformable-strap contacts are invisible to PhysX and are not in `forces`.
        # A soft bag body has no contact sensor (PhysX cannot report contacts on deformables): force stays 0.
        if self._contact is not None:
            forces = self._link_forces()
            on_lower = self._on_lower_body()
            penalised = forces * on_lower.float()
            self._jaw_off_handle_force = penalised[:, self._rigid_jaw_col] + penalised[:, self._moving_jaw_col]
            self._other_force = penalised[:, self._other_cols].sum(dim=-1)
            self._collision_force = self._other_force + self._jaw_off_handle_force

        # LIFT: height above where the bag settled (running minimum of its height this episode).
        bag_z = self._bag_pose()[0][:, 2]
        self._bag_rest_z = torch.minimum(self._bag_rest_z, bag_z)
        self._lift_amt = bag_z - self._bag_rest_z
        # Hand height above the tip's resting pinch height. Unlike the bag lift this pays as
        # soon as the hand moves up with the jaws closed, so lifting is easy to discover.
        self._hand_rise = self._ee_pos_w()[:, 2] - (self._bag_rest_z + self._pinch_target[:, 1])

        # FINAL POSE: joints 1-6 against the hand-posed pose. The gripper is left out: it stalls wherever the strap is.
        pose_err = (self.arm.data.joint_pos.torch - self._final_pose)[:, self._arm_joint_ids].abs()
        self._pose_max = pose_err.amax(dim=-1)
        self._pose_rms = pose_err.square().mean(dim=-1).sqrt()
        # APPROACH POSE: all 7 joints against grasp_closed (the pinch the demo already found).
        self._approach_rms = (self.arm.data.joint_pos.torch - self._grasp_pose).square().mean(dim=-1).sqrt()
        self._track_rms = (self.arm.data.joint_pos.torch - self._teacher).square().mean(dim=-1).sqrt()
        at_final_pose = self._pose_max < cfg.final_pose_tol
        lifted = self._grasped & (self._lift_amt > cfg.lift_height)
        jaw_fwd = quat_apply(self.arm.data.body_quat_w.torch[:, self._ee_idx], self._insert_axis)
        self._jaw_z = jaw_fwd[:, 2]
        self._ee_up = self._jaw_z.clamp(0.0, 1.0)
        # Scoop task: success is the bag lift_height up while the grip is still true. The full task still
        # wants return_home.
        if self.cfg.scoop_teacher or self.cfg.success_is_lift:
            self._success = self._grasped & (self._lift_amt > cfg.lift_height)
            at_final_pose = self._lift_amt > cfg.lift_height
        else:
            # SUCCESS from physical evidence: the bag only stays 5 cm up if the closed jaws are holding the strap.
            self._success = closed_cmd & near_strap & (self._lift_amt > cfg.lift_height) & at_final_pose
        self._first_success = self._success & ~self._ever_success
        self._time_to_success = torch.where(
            self._first_success, self._demo_step.float() * self.step_dt, self._time_to_success
        )
        self._ever_lifted |= lifted
        self._ever_at_pose |= at_final_pose
        self._ever_success |= self._success
        self._dist = dist
        self._reached = reached
        self._closed = self._gripper_closed_frac()
        self._ever_reached |= reached
        newly_grasped = self._grasped & ~self._ever_grasped
        self._time_to_grasp = torch.where(
            newly_grasped, self._demo_step.float() * self.step_dt, self._time_to_grasp
        )
        self._ever_grasped |= self._grasped
        self._ever_collided |= self._collision_force > 1e-3
        self._ee_up_max = torch.maximum(self._ee_up_max, self._ee_up * self._grasped.float())

    # -------------------------------------------------------------------------
    # ACTIONS: 7 numbers in [-1, 1]. Training: residual around the grasp_poses.json teacher
    # (zero = replay the demo). The pose tool disables that and increments the last target instead.
    # -------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-1.0, 1.0)
        limits = self.arm.data.soft_joint_pos_limits.torch
        if self.cfg.scoop_teacher:
            t = self._demo_step.float() * self.step_dt
            edges = torch.cumsum(self._path_s, dim=-1)
            idx = (t.unsqueeze(-1) >= edges).sum(dim=-1).clamp(max=self._path_s.shape[1] - 1)
            start = torch.zeros_like(t)
            has_prev = idx > 0
            start[has_prev] = edges[has_prev, idx[has_prev] - 1]
            end = edges.gather(1, idx.unsqueeze(1)).squeeze(1)
            blend = ((t - start) / (end - start).clamp(min=1e-3)).clamp(0.0, 1.0)
            # Joint targets are linear between IK knots (constant speed, no stop at each knot). After the last
            # segment the blend is 1, so the final pose is held.
            rows = torch.arange(self.num_envs, device=self.device)
            q0 = self._q_path[rows, idx]
            q1 = self._q_path[rows, idx + 1]
            self._teacher.copy_(q0 + blend.unsqueeze(-1) * (q1 - q0))
            targets = self._teacher + self.cfg.action_scale * self.actions
        elif self.cfg.use_demo_residual:
            idx = (self._demo_step.float() * self.cfg.demo_speed).long().clamp(max=self._demo_len - 1)
            self._teacher.copy_(self._demo_q[idx])
            # Over the last residual_fade_frac of the demo, shrink the residual from action_scale (20 deg) to
            # carry_action_scale (4 deg) so the arm can sit inside the 10 deg return_home box but still steer.
            phase = idx.float() / float(max(self._demo_len - 1, 1))
            fade = ((1.0 - phase) / max(self.cfg.residual_fade_frac, 1e-6)).clamp(0.0, 1.0)
            scale = self.cfg.action_scale * fade + self.cfg.carry_action_scale * (1.0 - fade)
            targets = self._teacher + scale.unsqueeze(-1) * self.actions
        else:
            # No teacher: each action is a joint step away from the pose being held, which starts as the shut grasp.
            targets = self._joint_targets + self.cfg.action_scale * self.actions
        if self.cfg.hold_gripper_closed:
            targets[:, self._grip_idx] = self.cfg.gripper_closed_pos
        self._demo_step += 1
        self._joint_targets = targets.clamp(limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self.arm.actuators.target_command.set_position_index(value=self._joint_targets)

    # -------------------------------------------------------------------------
    # OBSERVATIONS (47): joint pos rel. home (7), joint vel (7), joints -> teacher pose (7), demo phase (1),
    # tip->target (3), bag pos (3), bag quat (4), last actions (7), tip in the bag frame (3), and the grasp
    # state (5): rigid / moving jaw-to-strap gap, grasped, bag lift, gripper closed fraction.
    # If you add a term, bump cfg.observation_space. The strap gaps are sim-only (teacher policy).
    # -------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self.arm.data.joint_pos.torch
        joint_pos_rel = joint_pos - self.arm.data.default_joint_pos.torch
        joint_vel = self.arm.data.joint_vel.torch
        joint_track_err = joint_pos - self._teacher
        if self.cfg.scoop_teacher:
            total = self._path_s.sum(dim=-1).clamp(min=1e-3)
            phase = (self._demo_step.float() * self.step_dt / total).clamp(max=1.0).unsqueeze(-1)
        elif not self.cfg.use_demo_residual:
            phase = (
                self._demo_step.float() * self.step_dt / max(self.cfg.episode_length_s, 1e-3)
            ).clamp(max=1.0).unsqueeze(-1)
        else:
            demo_idx = (self._demo_step.float() * self.cfg.demo_speed).clamp(max=self._demo_len - 1)
            phase = (demo_idx / float(max(self._demo_len - 1, 1))).unsqueeze(-1)
        tip_to_target = self._target_pos_w() - self._ee_pos_w()
        bag_pos_w, bag_quat = self._bag_pose()
        bag_pos_local = bag_pos_w - self.scene.env_origins
        bag_quat = torch.where(bag_quat[:, 3:4] < 0.0, -bag_quat, bag_quat)  # q and -q are the same rotation
        tip_bag = quat_apply_inverse(bag_quat, self._ee_pos_w() - bag_pos_w)
        grasp_state = torch.stack(
            (
                self._rigid_gap.clamp(max=0.1),
                self._moving_gap.clamp(max=0.1),
                self._grasped.float(),
                self._lift_amt.clamp(-0.05, 0.3),
                self._closed,
            ),
            dim=-1,
        )
        obs = torch.cat(
            (
                joint_pos_rel,
                joint_vel,
                joint_track_err,
                phase,
                tip_to_target,
                bag_pos_local,
                bag_quat,
                self.actions,
                tip_bag,
                grasp_state,
            ),
            dim=-1,
        )
        return {"policy": obs}

    # -------------------------------------------------------------------------
    # DONES: first success ends the episode (time-to-success). Bag knocked away is a failure.
    # Hitting episode_length_s is a timeout. Also refreshes the task state that _get_rewards reads.
    # -------------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_state()
        import sys  # TMPDEBUG
        sys.path.insert(0, "/tmp")  # TMPDEBUG
        import pen_probe  # TMPDEBUG
        pen_probe.probe(self, int(self.episode_length_buf[0])); pen_probe.detail(self, int(self.episode_length_buf[0])); pen_probe.detail2(self, int(self.episode_length_buf[0])); pen_probe.shot(self, int(self.episode_length_buf[0])); pen_probe.onset(self, int(self.episode_length_buf[0])); pen_probe.bagprobe(self, int(self.episode_length_buf[0]))  # TMPDEBUG
        bag_xy = self._bag_pose()[0][:, :2] - self.scene.env_origins[:, :2]
        self._knocked_away = torch.norm(bag_xy, dim=-1) > self.cfg.bag_out_of_bounds
        terminated = (self._success if self.cfg.terminate_on_success else False) | self._knocked_away
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    # -------------------------------------------------------------------------
    # REWARDS. Per-step terms are scaled by step_dt; the success bonus is a flat one-off.
    # -------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        holding = self._grasped.float()
        # approach terms only pay before the grasp, so they stop pulling the hand back down during the carry
        approach = 1.0 - holding
        rew_reach = approach * ((1.0 - torch.tanh(self._dist / 0.20)) + (1.0 - torch.tanh(self._dist / 0.04)))
        rew_align = approach * self._insert_alignment() * (1.0 - torch.tanh(self._dist / 0.15))
        rew_reached = approach * self._reached.float()
        # Distance from the jaws to the strap itself. The goal point is where the tip sits before closing; the
        # closing sweep moves the tip ~4 cm off it, so "on the strap" is measured from the strap nodes.
        strap_gap = torch.minimum(self._rigid_gap, self._moving_gap) if self._deformable_handle else self._dist
        rew_close = (1.0 - torch.tanh(strap_gap / 0.01)) * self._closed
        # grasp, then lift progress (only while grasped)
        rew_grasp = holding
        rew_lift = holding * (self._lift_amt / cfg.lift_height).clamp(0.0, 1.0)
        # The hand only earns rise reward while a jaw stays at the strap. Otherwise the policy lifts the
        # hand off a strap it is not really holding (observed: hand rise 3 cm, bag lift 0).
        stays_on_strap = 1.0 - torch.tanh(strap_gap / cfg.carry_radius)
        rew_hand_up = holding * (self._hand_rise / cfg.lift_height).clamp(0.0, 1.0) * stays_on_strap
        rew_squeeze = holding * self._closed
        # Carry: rigid-jaw long axis (link -Y) toward world +Z. Paid only while grasped.
        rew_ee_up = holding * self._ee_up
        # follow the hand-posed sequence (residual 0 is the open-loop demo); return still pays once holding
        rew_track = 1.0 - torch.tanh(self._track_rms / cfg.track_scale)
        rew_return = holding * (1.0 - torch.tanh(self._pose_rms / cfg.return_scale))
        rew_residual = torch.sum(torch.square(self.actions), dim=-1)  # prefer riding the teacher when it works
        rew_collision = torch.tanh(self._collision_force / cfg.collision_force_scale)
        collision_ramp = min(1.0, float(self.common_step_counter) / max(cfg.collision_ramp_steps, 1))
        rew_action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)

        shaped = (
            cfg.rew_scale_reach * rew_reach
            + cfg.rew_scale_align * rew_align
            + cfg.rew_scale_reached * rew_reached
            + cfg.rew_scale_close * rew_close
            + cfg.rew_scale_track * rew_track
            + cfg.rew_scale_grasp * rew_grasp
            + cfg.rew_scale_squeeze * rew_squeeze
            + cfg.rew_scale_ee_up * rew_ee_up
            + cfg.rew_scale_hand_up * rew_hand_up
            + cfg.rew_scale_lift * rew_lift
            + cfg.rew_scale_return * rew_return
            + cfg.rew_scale_collision * collision_ramp * rew_collision
            + cfg.rew_scale_action_rate * rew_action_rate
            + cfg.rew_scale_residual * rew_residual
            + cfg.rew_scale_time
        )
        log = self.extras.setdefault("log", {})
        log["Metrics/collision_weight"] = cfg.rew_scale_collision * collision_ramp
        log["Metrics/tip_dist"] = self._dist.mean()
        log["Metrics/collision_force"] = self._collision_force.mean()
        log["Metrics/lift_height"] = self._lift_amt.mean()
        log["Metrics/hand_rise"] = self._hand_rise.mean()
        log["Metrics/final_pose_err_deg"] = torch.rad2deg(self._pose_max).mean()
        log["Metrics/approach_err_deg"] = torch.rad2deg(self._approach_rms).mean()
        log["Metrics/demo_track_err_deg"] = torch.rad2deg(self._track_rms).mean()
        log["Metrics/force_jaw_on_body"] = self._jaw_off_handle_force.mean()
        log["Metrics/force_other_links"] = self._other_force.mean()
        log["Metrics/ee_up"] = rew_ee_up.mean()
        log["Metrics/grasp_raw"] = self._grasp_raw.float().mean()
        log["Metrics/grasped"] = holding.mean()
        # One-off penalty when the bag is knocked away. Without it, ending the episode this way stops the living
        # cost, so a policy that has drifted off the teacher is paid to push the bag out of bounds.
        terminal = (
            cfg.rew_scale_success * self._first_success.float()
            + cfg.rew_scale_knocked_away * self._knocked_away.float()
        )
        return shaped * self.step_dt + terminal

    # -------------------------------------------------------------------------
    # RESET: report the rates of the episodes that just ended, then re-randomise them.
    # -------------------------------------------------------------------------
    def _report(self, env_ids) -> None:
        reached = self._ever_reached[env_ids].float()
        grasp = self._ever_grasped[env_ids].float()
        lift = self._ever_lifted[env_ids].float()  # grasped with the bag 5 cm up at some point
        final = self._ever_success[env_ids].float()  # grasped and at the final pose at some point this episode
        at_pose = self._ever_at_pose[env_ids].float()
        collided = self._ever_collided[env_ids].float()
        knocked = self._knocked_away[env_ids].float()

        log = self.extras.setdefault("log", {})  # -> TensorBoard
        log["Metrics/reach_rate"] = reached.mean()
        log["Metrics/grasp_rate"] = grasp.mean()
        log["Metrics/lift_rate"] = lift.mean()
        log["Metrics/final_pose_rate"] = final.mean()
        log["Metrics/at_final_pose_rate"] = at_pose.mean()
        log["Metrics/collision_rate"] = collided.mean()
        log["Metrics/knocked_away_rate"] = knocked.mean()
        succeeded = self._ever_success[env_ids]
        if succeeded.any():
            log["Metrics/time_to_success_s"] = self._time_to_success[env_ids][succeeded].mean()
        grasped_eps = self._ever_grasped[env_ids]
        if grasped_eps.any():
            log["Metrics/time_to_grasp_s"] = self._time_to_grasp[env_ids][grasped_eps].mean()
        log["Metrics/episode_length_s"] = self.episode_length_buf[env_ids].float().mean() * self.step_dt
        log["Metrics/rand_scale"] = self._rand_scale

        # -> terminal: one line per ~num_envs finished episodes (skip the construction-time reset)
        if int(self.common_step_counter) == 0:
            return
        self._win_count += len(env_ids)
        self._win_sums += torch.stack(
            (reached.sum(), grasp.sum(), lift.sum(), final.sum(), at_pose.sum(), collided.sum(), knocked.sum())
        )
        if succeeded.any():
            tts = self._time_to_success[env_ids][succeeded]
            self._win_tts_sum += float(tts.sum())
            self._win_tts_n += int(succeeded.sum())
        grasped_eps = self._ever_grasped[env_ids]
        if grasped_eps.any():
            tts_g = self._time_to_grasp[env_ids][grasped_eps]
            self._win_grasp_tts_sum += float(tts_g.sum())
            self._win_grasp_tts_n += int(grasped_eps.sum())
        # Still grasped when the episode ended. For the scoop task this is the success rate.
        # Grip false is the raw geometric test. The debounced flag stays true for half a second after it.
        self._win_held_sum += float(self._grasp_raw[env_ids].float().sum())
        self._win_ee_sum += float(self._ee_up_max[env_ids].sum())
        if self._win_count >= self.num_envs:
            reach_r, grasp_r, lift_r, final_r, pose_r, coll_r, knock_r = (self._win_sums / self._win_count).tolist()
            if self.cfg.report_focus == "grasp":
                tts_txt = (
                    f"{self._win_grasp_tts_sum / self._win_grasp_tts_n:.1f} s" if self._win_grasp_tts_n else "n/a"
                )
                print(
                    f"[{self.cfg.report_name}] episodes: {self._win_count:4d} | grasp rate: {grasp_r:.3f} | "
                    f"collision rate: {coll_r:.3f} | time to grasp: {tts_txt}",
                    flush=True,
                )
            elif self.cfg.report_focus == "lift":
                held_r = self._win_held_sum / self._win_count
                ee_r = self._win_ee_sum / self._win_count
                succ_r = self._win_tts_n / self._win_count
                print(
                    f"[{self.cfg.report_name}] episodes: {self._win_count:4d} | held rate: {held_r:.3f} | "
                    f"dropped: {1.0 - held_r:.3f} | lift ({self.cfg.lift_height * 100:.0f} cm): {lift_r:.3f} | "
                    f"ee up: {ee_r:.3f} | success: {succ_r:.3f} | collision rate: {coll_r:.3f}",
                    flush=True,
                )
            else:
                tts_txt = f"{self._win_tts_sum / self._win_tts_n:.1f} s" if self._win_tts_n else "n/a"
                print(
                    f"[{self.cfg.report_name}] episodes: {self._win_count:4d} | reach rate: {reach_r:.3f} | "
                    f"grasp rate: {grasp_r:.3f} | lift ({self.cfg.lift_height * 100:.0f} cm) rate: {lift_r:.3f} | "
                    f"final-pose rate: {final_r:.3f} (at pose {pose_r:.3f}) | collision rate: {coll_r:.3f} | "
                    f"knocked away: {knock_r:.3f} | "
                    f"time to success: {tts_txt} | bag randomization: {self._rand_scale:.1f}",
                    flush=True,
                )
                if self.cfg.rand_curriculum and final_r >= self.cfg.rand_curriculum_success and self._rand_scale < 1.0:
                    self._rand_scale = min(1.0, self._rand_scale + self.cfg.rand_scale_step)
                    print(
                        f"[{self.cfg.report_name}] curriculum: bag randomization scale -> {self._rand_scale:.1f}",
                        flush=True,
                    )
            self._win_count = 0
            self._win_sums.zero_()
            self._win_tts_sum = 0.0
            self._win_tts_n = 0
            self._win_grasp_tts_sum = 0.0
            self._win_grasp_tts_n = 0
            self._win_held_sum = 0.0
            self._win_ee_sum = 0.0

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _write_arm(self, env_ids: torch.Tensor, q: torch.Tensor) -> None:
        """Set joint positions for ``env_ids`` and refresh kinematics. ``q`` is (len(env_ids), 7)."""
        self.arm.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.sim.forward()

    def _wrist_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.arm.data.body_pos_w.torch[env_ids, self._ee_idx],
            self.arm.data.body_quat_w.torch[env_ids, self._ee_idx],
        )

    def _ensure_mouth_frame(self) -> None:
        """Mouth direction and the face point on the rigid jaw, in the wrist frame. Depends only on the closed jaw."""
        if self._mouth_local is not None:
            return
        saved = self.arm.data.joint_pos.torch[0:1].clone()
        q = self._grasp_pose.unsqueeze(0).clone()
        q[0, self._grip_idx] = self.cfg.gripper_closed_pos
        self._write_arm(saved.new_zeros(1, dtype=torch.long), q)  # env 0, restored below
        rigid_w, moving_w = self._jaw_points_w()
        mouth_w = moving_w[0].mean(dim=0) - rigid_w[0].mean(dim=0)
        wrist_q = self.arm.data.body_quat_w.torch[0, self._ee_idx]
        mouth = self._unit(quat_apply_inverse(wrist_q.unsqueeze(0), mouth_w.unsqueeze(0))[0])
        y = self._jaw_pts_rigid[:, 1]
        depth = (y - self.cfg.mouth_base_y) / (self.cfg.mouth_tip_y - self.cfg.mouth_base_y)
        dots = self._jaw_pts_rigid @ mouth
        dots = torch.where((depth > 0.35) & (depth < 0.75), dots, torch.full_like(dots, -1e9))
        face = self._jaw_pts_rigid[int(dots.argmax())].clone()
        moving_local = quat_apply_inverse(
            wrist_q.unsqueeze(0).expand(moving_w.shape[1], 4), moving_w[0] - self.arm.data.body_pos_w.torch[0, self._ee_idx]
        )
        # The moving-jaw point nearest the rigid face is the other side of the pinch.
        moving_face = moving_local[int((moving_local - face).norm(dim=-1).argmin())].clone()
        self._mouth_local = mouth
        self._face_local = face
        self._moving_face_local = moving_face
        self._write_arm(torch.zeros(1, dtype=torch.long, device=self.device), saved)

    def _quat_mapping(
        self, local_a: torch.Tensor, local_b: torch.Tensor, world_a: torch.Tensor, world_b: torch.Tensor
    ) -> torch.Tensor:
        """Rotation sending ``local_a`` / ``local_b`` onto ``world_a`` / ``world_b`` (batched world vectors)."""

        def basis(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x = self._unit(a)
            y = b - (b * x).sum(dim=-1, keepdim=True) * x
            y = self._unit(y)
            z = torch.cross(x, y, dim=-1)
            return torch.stack((x, y, z), dim=-1)  # columns

        n = world_a.shape[0]
        left = basis(local_a.expand(n, 3), local_b.expand(n, 3))
        right = basis(world_a, world_b)
        return quat_from_matrix(right @ left.transpose(-1, -2))

    def _bag_axis(self, bag_pose: torch.Tensor, axis: tuple[float, float, float]) -> torch.Tensor:
        q_bag = bag_pose[:, 3:7]
        local = q_bag.new_tensor(axis).view(1, 3).expand(q_bag.shape[0], 3)
        return quat_apply(q_bag, local)

    def _grasp_wrist_target(self, bag_pose: torch.Tensor, roll: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrist pose that pinches the strap from above. ``bag_pose`` is (n, 7).

        The jaw points down the hole (bag -Y, world -Z while the bag lies flat) and ``roll`` spins the
        mouth about that axis. The midpoint of the two jaw faces lands on the strap apex.
        """
        n = bag_pose.shape[0]
        apex_local = self._face_local.new_tensor(self.cfg.strap_apex).view(1, 3).expand(n, 3)
        apex = bag_pose[:, :3] + quat_apply(bag_pose[:, 3:7], apex_local)
        approach = -self._bag_axis(bag_pose, (0.0, 1.0, 0.0))
        pinch = math.cos(roll) * self._bag_axis(bag_pose, (0.0, 0.0, 1.0)) + math.sin(roll) * self._bag_axis(
            bag_pose, (1.0, 0.0, 0.0)
        )
        quat = self._quat_mapping(self._insert_local, self._mouth_local, approach, pinch)
        mid = 0.5 * (self._face_local + self._moving_face_local)
        wrist = apex - quat_apply(quat, mid.expand(n, 3))
        return wrist, quat

    def _recorded_wrist_target(self, bag_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Same pinch point, with the jaw orientation recorded in grasp_goal.json. None when there is no recording."""
        if self._goal_quat is None:
            return None
        n = bag_pose.shape[0]
        quat = quat_mul(bag_pose[:, 3:7], self._goal_quat[:n])
        apex_local = self._face_local.new_tensor(self.cfg.strap_apex).view(1, 3).expand(n, 3)
        apex = bag_pose[:, :3] + quat_apply(bag_pose[:, 3:7], apex_local)
        mid = 0.5 * (self._face_local + self._moving_face_local)
        wrist = apex - quat_apply(quat, mid.expand(n, 3))
        return wrist, quat

    def _ik(
        self,
        env_ids: torch.Tensor,
        pos_des: torch.Tensor,
        quat_des: torch.Tensor,
        seed: torch.Tensor,
        nominal: torch.Tensor | None = None,
        posture_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Levenberg-Marquardt IK for the wrist, biased toward ``nominal``.

        This is the update MoveIt's LMA kinematics solver uses: hit the wrist pose, but each step is
        pulled toward the seed so the arm stays in that posture instead of flipping into another
        solution of the same pose. ``posture_weight`` scales that pull per arm joint. The gripper
        joint stays closed. Returns (n, 7).
        """
        q = seed.clone()
        q[:, self._grip_idx] = self.cfg.gripper_closed_pos
        if nominal is None:
            nominal = q
        limits = self.arm.data.soft_joint_pos_limits.torch[env_ids]
        eye = torch.eye(6, device=self.device)
        arm_cols = self._arm_jac_cols
        arm_ids = self._arm_joint_ids
        lam2 = 0.05**2
        if posture_weight is None:
            posture_weight = torch.ones(q.shape[0], len(arm_ids), device=self.device)
        for _ in range(40):
            self._write_arm(env_ids, q)
            pos, quat = self._wrist_pose(env_ids)
            quat_goal = torch.where((quat_des * quat).sum(dim=-1, keepdim=True) < 0.0, -quat_des, quat_des)
            pos_err = pos_des - pos
            rot_err = quat_box_minus(quat_goal, quat)
            rot_norm = rot_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            if float(pos_err.norm(dim=-1).max()) < 0.004 and float(rot_norm.max()) < 0.05:
                break
            pos_err = pos_err.clamp(-0.02, 0.02)
            rot_err = rot_err * (0.25 / rot_norm.clamp(min=0.25))
            jac = self.arm.data.body_link_jacobian_w.torch[env_ids, self._jacobi_ee][:, :, arm_cols]
            dx = torch.cat((pos_err, rot_err), dim=-1)
            # (J^T J + λ^2 I) dq = J^T dx + λ^2 W (q_nominal - q). W is 1, except on a joint the
            # caller wants held to the nominal (the elbow, while the wrist still aims the jaw).
            jt = jac.transpose(-1, -2)
            posture = (nominal[:, arm_ids] - q[:, arm_ids]).clamp(-0.5, 0.5) * posture_weight
            dq = torch.linalg.solve(jt @ jac + lam2 * eye, jt @ dx.unsqueeze(-1) + lam2 * posture.unsqueeze(-1))
            dq = dq.squeeze(-1).clamp(-0.2, 0.2)
            lo = limits[:, :, 0][:, arm_ids]
            hi = limits[:, :, 1][:, arm_ids]
            q[:, arm_ids] = (q[:, arm_ids] + dq).clamp(lo, hi)
            q[:, self._grip_idx] = self.cfg.gripper_closed_pos
        return q

    def _geometry_holds(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Debounce-free grasp test for ``env_ids``: closed command is separate; this is the strap in the jaws."""
        cfg = self.cfg
        resting = (self._rest_count[env_ids] >= cfg.min_strap_nodes) & (
            self._rest_depth[env_ids] < cfg.mouth_mid_frac + cfg.mouth_tol
        )
        pinched = (self._strap_count[env_ids] >= cfg.min_strap_nodes) & (
            (self._strap_depth[env_ids] - cfg.mouth_mid_frac).abs() < cfg.mouth_tol
        )
        return resting | pinched

    def _load_scoop_trajectory(self) -> None:
        """Read cfg.scoop_trajectory_file: absolute waypoints from the spawned grasp (see the file's _comment)."""
        with open(self.cfg.scoop_trajectory_file) as f:
            spec = json.load(f)
        self._scoop_hold_s = float(spec.get("hold_s", 2.0))
        self._scoop_wps = []
        for i, w in enumerate(spec["waypoints"]):
            wp = {
                "name": str(w.get("name", f"waypoint {i + 1}")),
                "away": float(w.get("away_cm", 0.0)) / 100.0,
                "up": float(w.get("up_cm", 0.0)) / 100.0,
                "jaw_deg": None if w.get("jaw_deg") is None else float(w["jaw_deg"]),
                "seconds": float(w.get("seconds", 1.0)),
                "knots": max(int(w.get("knots", 3)), 1),
                "rotate_frac": min(max(float(w.get("rotate_frac", 1.0)), 0.05), 1.0),
            }
            if wp["seconds"] <= 0.0:
                raise ValueError(f"{self.cfg.scoop_trajectory_file}: waypoint '{wp['name']}' needs seconds > 0")
            self._scoop_wps.append(wp)
        if not self._scoop_wps:
            raise ValueError(f"{self.cfg.scoop_trajectory_file} has no waypoints")

    def _build_scoop_teacher(self, env_ids: torch.Tensor, q_grasp: torch.Tensor, bag_pose: torch.Tensor) -> None:
        """Solve the waypoint trajectory in ``cfg.scoop_trajectory_file`` by IK from the spawned closed grasp.

        Everything happens in the vertical plane spanned by ``away`` (horizontal, from the bag body toward the
        strap) and ``up`` (world +Z). The tracked point is the rigid-jaw tip. The jaw elevation is the angle of
        the jaw's long axis above the horizontal ``away`` direction: -90 deg points straight down, 0 is parallel to
        the ground, +90 is straight up.

        Waypoints are absolute offsets from the spawned grasp. Each move starts from the pose the arm actually reached
        at the end of the previous one (so a missed waypoint never makes the next move jump back). Each move is cut into straight IK knots in (tip,
        elevation) (the jaw finishes turning after the first ``rotate_frac`` of the move), every knot seeded from the previous one, with the jaw's roll about its own axis kept from the
        spawned grasp. The joint targets are linear between knots, over the waypoint's ``seconds`` or the time the
        fastest joint needs, whichever is longer: ``scoop_joint_speed`` for joints 1-3 and the higher
        ``scoop_wrist_speed`` for the wrist joints 4-6 that turn the jaw. Zero action replays this path, then holds
        the last pose for ``hold_s``.
        """
        cfg = self.cfg
        n = q_grasp.shape[0]
        q0 = q_grasp.clone()
        q0[:, self._grip_idx] = cfg.gripper_closed_pos
        self._write_arm(env_ids, q0)
        tip0 = self._ee_pos_w()[env_ids]
        _, quat0 = self._wrist_pose(env_ids)
        offset = self._ee_offset[env_ids]

        up = torch.zeros(n, 3, device=self.device)
        up[:, 2] = 1.0
        away = self._bag_axis(bag_pose, (0.0, 0.0, 1.0))  # bag +Z: from the body toward the strap
        away = self._unit(away - (away * up).sum(dim=-1, keepdim=True) * up)
        axis = torch.cross(away, up, dim=-1)  # turning about this raises the jaw elevation

        def elevation(jaw_w: torch.Tensor) -> torch.Tensor:
            jaw_w = self._unit(jaw_w)
            return torch.atan2((jaw_w * up).sum(dim=-1), (jaw_w * away).sum(dim=-1))

        elev0 = elevation(quat_apply(quat0, self._insert_local.expand(n, 3)))
        stages = []
        elev_prev = elev0
        for wp in self._scoop_wps:
            tip_w = tip0 + wp["away"] * away + wp["up"] * up
            elev_w = elev_prev if wp["jaw_deg"] is None else torch.full_like(elev0, math.radians(wp["jaw_deg"]))
            stages.append((tip_w, elev_w))
            elev_prev = elev_w

        q = q0
        k = 0
        self._q_path[env_ids, 0] = q0
        tip_a, elev_a = tip0, elev0
        lines = []
        stage_end_tip = tip0
        # Pull toward the previous knot's posture. Near the spawned grasp the wrist is almost singular (joint 5 ~ 0),
        # and a strong pull stalls the IK there with a 1-9 cm tip error, so the default is no pull.
        posture = torch.full((n, len(self._arm_joint_ids)), cfg.scoop_posture_weight, device=self.device)
        for s, (tip_b, elev_b) in enumerate(stages):
            n_knots = int(self._scoop_wps[s]["knots"])
            seg_s = float(self._scoop_wps[s]["seconds"]) / n_knots
            worst_mm, worst_deg = 0.0, 0.0
            peak_arm, peak_wrist = 0.0, 0.0
            stage_start_tip = stage_end_tip
            for j in range(1, n_knots + 1):
                f = j / n_knots
                tip = tip_a + f * (tip_b - tip_a)
                f_rot = min(1.0, f / float(self._scoop_wps[s]["rotate_frac"]))  # jaw turns early, tip keeps rising
                quat = quat_mul(quat_from_angle_axis(elev_a + f_rot * (elev_b - elev_a) - elev0, axis), quat0)
                wrist = tip - quat_apply(quat, offset)
                q_new = self._ik(env_ids, wrist, quat, q, q, posture_weight=posture)
                self._write_arm(env_ids, q_new)  # _ik leaves the arm one iterate behind; measure the real miss
                pos, rot = self._wrist_pose(env_ids)
                goal = torch.where((quat * rot).sum(dim=-1, keepdim=True) < 0.0, -quat, quat)
                worst_mm = max(worst_mm, float((wrist - pos).norm(dim=-1).max()) * 1000.0)
                worst_deg = max(worst_deg, float(quat_box_minus(goal, rot).norm(dim=-1).max()) * 180.0 / math.pi)
                dq = (q_new - q).abs()
                dq_arm = dq[:, self._shoulder_ids].amax(dim=-1)
                dq_wrist = dq[:, self._wrist_ids].amax(dim=-1)
                knot_s = torch.clamp(
                    torch.maximum(dq_arm / cfg.scoop_joint_speed, dq_wrist / cfg.scoop_wrist_speed), min=seg_s
                )
                self._path_s[env_ids, k] = knot_s
                peak_arm = max(peak_arm, float((dq_arm / knot_s).max()))
                peak_wrist = max(peak_wrist, float((dq_wrist / knot_s).max()))
                k += 1
                self._q_path[env_ids, k] = q_new
                q = q_new
            stage_end_tip = self._ee_pos_w()[env_ids]
            _, rot = self._wrist_pose(env_ids)
            got = stage_end_tip - stage_start_tip
            e_deg = math.degrees(float(elevation(quat_apply(rot, self._insert_local.expand(n, 3))).mean()))
            lines.append(
                f"  [{s + 1}] {self._scoop_wps[s]['name']}: tip moved {float((got * away).sum(-1).mean()) * 1000:+.1f} mm away, "
                f"{float((got * up).sum(-1).mean()) * 1000:+.1f} mm up | jaw elevation {e_deg:+.1f} deg | "
                f"worst IK miss {worst_mm:.1f} mm, {worst_deg:.1f} deg | peak joint speed: arm {peak_arm:.2f}, "
                f"wrist {peak_wrist:.2f} rad/s"
            )
            if worst_mm > cfg.scoop_ik_warn_mm or worst_deg > cfg.scoop_ik_warn_deg:
                lines[-1] += "  <-- the arm cannot reach this stage exactly; the teacher will lag it"
            # The next move starts where the arm really ended, not at the commanded target. If the IK missed this
            # waypoint, starting the next line at the unreached target would make the first knot jump back to it.
            tip_a, elev_a = stage_end_tip, elevation(quat_apply(rot, self._insert_local.expand(n, 3)))

        duration = float(self._path_s[env_ids].sum(dim=-1).max())
        need = duration + self._scoop_hold_s
        cfg.episode_length_s = need if not self._spawn_announced else max(cfg.episode_length_s, need)
        if not self._spawn_announced:
            print(
                f"[{cfg.report_name}] scoop teacher: jaw elevation {math.degrees(float(elev0.mean())):+.1f} deg at the "
                f"grasp, path {duration:.1f} s + {self._scoop_hold_s:.1f} s hold, episode {cfg.episode_length_s:.1f} s\n"
                + "\n".join(lines),
                flush=True,
            )
        else:
            for line in lines:
                if "<--" in line:
                    print(f"[{cfg.report_name}] scoop teacher{line.rstrip()}", flush=True)

    def _grasp_target(self, bag_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrist target for the roll chosen on the first spawn. ``bag_pose`` is (n, 7)."""
        if self._pinch_roll is None:
            recorded = self._recorded_wrist_target(bag_pose)
            if recorded is not None:
                return recorded
            self._pinch_roll = 0.0
        return self._grasp_wrist_target(bag_pose, self._pinch_roll)

    def _score_grasp(
        self, env_ids: torch.Tensor, pos_des: torch.Tensor, q: torch.Tensor, nominal: torch.Tensor
    ) -> tuple[tuple[float, float, float], float]:
        """Rank a solved grasp: strap held, then closeness to the comfortable seed, then wrist error."""
        self._write_arm(env_ids, q)
        self._strap_in_jaws()
        held = float(self._geometry_holds(env_ids).float().mean())
        perr = float((self._wrist_pose(env_ids)[0] - pos_des).norm(dim=-1).mean())
        posture = float((q[:, self._arm_joint_ids] - nominal[:, self._arm_joint_ids]).pow(2).mean())
        # A pose that misses the wrist by more than 2 cm is not on the strap, even if a proxy point is close.
        # Ties that do hold the strap go to the arm closest to the recorded posture.
        grasped = held if perr < 0.02 else 0.0
        return (grasped, 0.0 if grasped > 0.0 else -perr, -posture), perr

    def _spawn_closed_grasp(self, env_ids, bag_pose: torch.Tensor) -> None:
        """Put the closed gripper on ``bag_pose``'s strap and, when asked, build the upward teacher.

        ``bag_pose`` rows match ``env_ids``. The jaw points down the hole. The roll about that axis is the
        one MoveIt's compute_ik would keep: the solution nearest the recorded arm posture that still holds
        the strap. The jaws are written shut; nothing closes during the episode.
        """
        if not self._deformable_handle:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._ensure_mouth_frame()
        n = bag_pose.shape[0]
        nominal = self._grasp_nominal.unsqueeze(0).expand(n, -1).clone()
        nominal[:, self._grip_idx] = self.cfg.gripper_closed_pos
        if self._pinch_roll is None and self._goal_quat is None:
            self._pinch_roll = 0.0
        if not self._spawn_announced:
            # First spawn picks the roll. Later resets reuse it, including after the bag pose changes:
            # the roll is in the bag frame, and the IK below follows each bag.
            probe, pose0, seed0 = env_ids[:1], bag_pose[:1], nominal[:1]
            candidates: list[tuple[float | None, tuple[torch.Tensor, torch.Tensor]]] = []
            recorded = self._recorded_wrist_target(pose0)
            if recorded is not None:
                candidates.append((None, recorded))
            candidates.extend(
                (i * math.pi / 4.0, self._grasp_wrist_target(pose0, i * math.pi / 4.0)) for i in range(8)
            )
            best_key: tuple[float, float, float] | None = None
            best_roll: float | None = 0.0
            for roll, (pos, quat) in candidates:
                q = self._ik(probe, pos, quat, seed0, seed0)
                key, _ = self._score_grasp(probe, pos, q, seed0)
                if best_key is None or key > best_key:
                    best_key, best_roll = key, roll
            self._pinch_roll = best_roll
        pos, quat = self._grasp_target(bag_pose)
        best_q = self._ik(env_ids, pos, quat, nominal, nominal)
        if self.cfg.spawn_press > 0.0:
            press_pos = pos - self.cfg.spawn_press * self._unit(quat_apply(quat, self._mouth_local.expand(n, 3)))
            pressed = self._ik(env_ids, press_pos, quat, best_q, nominal)
            _, perr = self._score_grasp(env_ids, press_pos, pressed, nominal)
            if perr < 0.008:
                best_q = pressed
                pos = press_pos
        if self.cfg.scoop_teacher:
            self._build_scoop_teacher(env_ids, best_q, bag_pose)
        self._write_arm(env_ids, best_q)
        self._strap_in_jaws()
        self._q_grasp[env_ids] = best_q
        self._joint_targets[env_ids] = best_q
        self._teacher[env_ids] = best_q
        held = self._geometry_holds(env_ids)
        self._grasped[env_ids] = held
        self._grasp_raw[env_ids] = held
        self._grasp_streak[env_ids] = torch.where(
            held, torch.full_like(self._grasp_streak[env_ids], self.cfg.grasp_on_steps), torch.zeros_like(self._grasp_streak[env_ids])
        )
        self._ever_grasped[env_ids] = held
        wrist, wrist_q = self._wrist_pose(env_ids)
        quat_des = torch.where((quat * wrist_q).sum(-1, keepdim=True) < 0.0, -quat, quat)
        if not self._spawn_announced or float(held.float().mean()) < 1.0:
            pos_mm = float((pos - wrist).norm(dim=-1).mean()) * 1000.0
            ori_deg = float(quat_box_minus(quat_des, wrist_q).norm(dim=-1).mean()) * 180.0 / math.pi
            approach = -self._bag_axis(bag_pose, (0.0, 1.0, 0.0))
            insert_w = self._unit(quat_apply(wrist_q, self._insert_local.expand(n, 3)))
            down_deg = float(torch.acos((insert_w * approach).sum(-1).clamp(-1.0, 1.0)).mean()) * 180.0 / math.pi
            climb = getattr(self, "_climb_mm", 0.0)
            turned = getattr(self, "_jaw_turn_deg", 0.0)
            print(
                f"[{self.cfg.report_name}] spawned closed grasp on {int(held.sum())}/{held.numel()} envs "
                f"(wrist err {pos_mm:.1f} mm, {ori_deg:.1f} deg, jaw {down_deg:.0f} deg off the hole, "
                f"climb {climb:.0f} mm in {getattr(self, '_climb_s', 0.0):.1f} s, jaw turn {turned:.0f} deg)",
                flush=True,
            )
            self._spawn_announced = True

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.arm._ALL_INDICES
        self._report(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        self._demo_step[env_ids] = 0
        if self.cfg.stagger_first_episodes and int(self.common_step_counter) == 0:
            # Every env starts at step 0 and would otherwise time out on the same step, so each PPO rollout
            # (horizon_length << episode length) would hold one slice of the demo only. Truncating the first
            # episodes at random lengths spreads the envs over the whole demo. The teacher still starts at home.
            self.episode_length_buf[env_ids] = torch.randint_like(
                self.episode_length_buf[env_ids], high=max(int(self.max_episode_length), 1)
            )

        # arm: the recorded grasp when this policy starts there, otherwise home, plus small jitter.
        # A closed-grasp spawn overwrites this with the jaws on the strap.
        if self._reset_pose is not None:
            joint_pos = self._reset_pose.unsqueeze(0).expand(n, -1).clone()
        else:
            joint_pos = self.arm.data.default_joint_pos.torch[env_ids].clone()
        if not self.cfg.spawn_closed_grasp:
            joint_pos += sample_uniform(-self.cfg.arm_joint_noise, self.cfg.arm_joint_noise, joint_pos.shape, self.device)
        limits = self.arm.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        self.arm.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)

        # bag: random xy, flip, tilt, heading (world composition: yaw * tilt * flip * lying-flat default),
        # with the ranges scaled by the randomization curriculum
        s = self._rand_scale
        if self._deformable_body:
            # a soft body has no root pose: start from the configured one (env-local)
            bag_pose = torch.cat((self._body_init_pos, self._body_init_quat)).expand(n, 7).clone()
        else:
            bag_pose = self.bag.data.default_root_pose.torch[env_ids].clone()  # [x, y, z, qx, qy, qz, qw], env-local
        xy_lo, xy_hi = self.cfg.bag_xy_range
        bag_pose[:, 0:2] += sample_uniform(s * xy_lo, s * xy_hi, (n, 2), self.device)
        zeros = torch.zeros(n, device=self.device)
        flip = (torch.rand(n, device=self.device) < 0.5).float() if self.cfg.bag_random_flip else zeros
        q_flip = quat_from_euler_xyz(flip * math.pi, zeros, zeros)
        tilt = s * self.cfg.bag_tilt_range
        roll = sample_uniform(-tilt, tilt, (n,), self.device)
        pitch = sample_uniform(-tilt, tilt, (n,), self.device)
        q_tilt = quat_from_euler_xyz(roll, pitch, zeros)
        yaw_lo, yaw_hi = self.cfg.bag_yaw_range
        yaw = sample_uniform(s * yaw_lo, s * yaw_hi, (n,), self.device)
        q_yaw = quat_from_euler_xyz(zeros, zeros, yaw)
        bag_pose[:, 3:7] = quat_mul(q_yaw, quat_mul(q_tilt, quat_mul(q_flip, bag_pose[:, 3:7])))
        bag_pose[:, 2] += 0.12 * torch.sin(
            torch.maximum(roll.abs(), pitch.abs())
        )  # tilted corner starts above the floor
        bag_pose[:, :3] += self.scene.env_origins[env_ids]
        if self._deformable_body:
            v = self._body_rest.shape[0]
            quat = bag_pose[:, 3:7].unsqueeze(1).expand(n, v, 4).reshape(-1, 4)
            local = self._body_rest.expand(n, v, 3).reshape(-1, 3)
            body_nodes = bag_pose[:, :3].unsqueeze(1) + quat_apply(quat, local).reshape(n, v, 3)
            self.bag.write_nodal_pos_to_sim_index(body_nodes, env_ids=env_ids)
            self.bag.write_nodal_velocity_to_sim_index(torch.zeros_like(body_nodes), env_ids=env_ids)
        else:
            self.bag.write_root_pose_to_sim_index(root_pose=bag_pose, env_ids=env_ids)
            self.bag.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids
            )

        # deformable strap: lay it onto the bag's new pose, at rest (its feet stay clamped to the body)
        if self._deformable_handle:
            v = self._handle_rest.shape[0]
            quat = bag_pose[:, 3:7].unsqueeze(1).expand(n, v, 4).reshape(-1, 4)
            local = self._handle_rest.expand(n, v, 3).reshape(-1, 3)
            nodes = bag_pose[:, :3].unsqueeze(1) + quat_apply(quat, local).reshape(n, v, 3)
            self.handle.write_nodal_pos_to_sim_index(nodes, env_ids=env_ids)
            self.handle.write_nodal_velocity_to_sim_index(torch.zeros_like(nodes), env_ids=env_ids)

        # task state: resting height starts at the spawn height and drops to where the bag settles
        self._bag_rest_z[env_ids] = bag_pose[:, 2]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = joint_pos
        self._teacher[env_ids] = self._demo_q[0]
        self._success[env_ids] = False
        self._knocked_away[env_ids] = False
        self._first_success[env_ids] = False
        self._time_to_success[env_ids] = 0.0
        self._time_to_grasp[env_ids] = 0.0
        self._grasped[env_ids] = False
        self._grasp_raw[env_ids] = False
        self._grasp_streak[env_ids] = 0
        self._lift_amt[env_ids] = 0.0
        self._rigid_gap[env_ids] = float("inf")
        self._moving_gap[env_ids] = float("inf")
        self._ever_reached[env_ids] = False
        self._ever_grasped[env_ids] = False
        self._ever_lifted[env_ids] = False
        self._ever_at_pose[env_ids] = False
        self._ever_success[env_ids] = False
        self._ever_collided[env_ids] = False
        self._ee_up[env_ids] = 0.0
        self._ee_up_max[env_ids] = 0.0
        if self.cfg.spawn_closed_grasp:
            self._spawn_closed_grasp(env_ids, bag_pose)

    # -------------------------------------------------------------------------
    # DEBUG VIS (Kit window only): red = rigid tip, green = pinch target on the strap
    # -------------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):

                def sphere(radius, color):
                    return sim_utils.SphereCfg(
                        radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                    )

                self._goal_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/BagLift/Points",
                        markers={
                            "tip": sphere(0.008, (1.0, 0.15, 0.1)),
                            "goal": sphere(0.01, (0.15, 1.0, 0.2)),
                        },
                    )
                )
            self._goal_visualizer.set_visibility(True)
        elif hasattr(self, "_goal_visualizer"):
            self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_goal_visualizer"):
            return
        idx = torch.arange(2 * self.num_envs, device=self.device) // self.num_envs  # 0=tip, 1=goal
        self._goal_visualizer.visualize(
            translations=torch.cat((self._ee_pos_w(), self._target_pos_w()), dim=0),
            marker_indices=idx,
        )
