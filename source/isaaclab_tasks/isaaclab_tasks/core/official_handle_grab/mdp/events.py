# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events: phase-balanced resets from the physics-validated dataset, encoder calibration bias, and the camera
task's camera-mount and lighting variation.

With the deformable strap, a prestartup event also ties the strap's feet to the bag body.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from isaaclab_tasks.core.official_handle_grab.scene import handle_grab_scene as scene_defaults

from ..reset.curriculum import ALL_PHASES, HOME, PHASES
from ..reset.dataset import DEFAULT_DATASET_PATH, ResetDataset, sample_rows, validate_phase_weights
from .task import BAG, ROBOT, strap_rest_nodes, write_strap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def attach_strap(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Clamp the deformable strap's two feet to the bag body in every environment, before the simulation starts.

    Authors two kinds of USD prims per environment, from the OmniPhysics deformable schemas PhysX reads at start:

    * ``HandleAttachment`` (``OmniPhysicsVtxXformAttachment``) ties the strap nodes in ``anchor_vertices`` (the bottom
      of each leg, which the ribbon extends 4 mm into the body) to the bag's frame, with infinite stiffness.
    * ``HandleCollisionFilter_<k>`` (``OmniPhysicsElementCollisionFilter``) stops the triangles around those nodes from
      colliding with the body, which they would otherwise fight.

    The strap is copied per environment as plain USD, so these relationships cannot be cloned and are written for
    every environment. The node order is the vertex order of ``assets/bag_handle_ribbon.usda``.
    """
    from pxr import Sdf, Usd, UsdPhysics, Vt

    info = json.loads(Path(scene_defaults.HANDLE_INFO_JSON).read_text())
    anchors = [int(i) for i in info["anchor_vertices"]]
    local = np.asarray(info["rest_points"], dtype=np.float32)[anchors]
    triangles = [int(i) for i in info["anchor_triangles"]]

    stage = env.scene.stage
    for env_path in env.scene.env_prim_paths:
        bag = stage.GetPrimAtPath(f"{env_path}/Bag")
        sim_mesh = stage.GetPrimAtPath(f"{env_path}/Handle/sim_mesh")
        if not bag.IsValid() or not sim_mesh.IsValid():
            raise RuntimeError(
                f"Cannot clamp the deformable strap in {env_path}: {env_path}/Bag or {env_path}/Handle/sim_mesh is"
                " missing. Check HANDLE_CFG in handle_grab_scene.py."
            )
        attachment = stage.DefinePrim(f"{env_path}/HandleAttachment", "OmniPhysicsVtxXformAttachment")
        attachment.CreateRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
        attachment.CreateRelationship("omniphysics:src1").SetTargets([bag.GetPath()])
        attachment.CreateAttribute("omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(anchors))
        attachment.CreateAttribute("omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray).Set(
            Vt.Vec3fArray.FromNumpy(local)
        )
        attachment.CreateAttribute("omniphysics:stiffness", Sdf.ValueTypeNames.Float).Set(float("inf"))

        colliders = [p for p in Usd.PrimRange(bag, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        for k, collider in enumerate(colliders):
            filtered = stage.DefinePrim(f"{env_path}/HandleCollisionFilter_{k}", "OmniPhysicsElementCollisionFilter")
            filtered.CreateRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
            filtered.CreateRelationship("omniphysics:src1").SetTargets([collider.GetPath()])
            # Group 0 is the strap's foot triangles; the body has no groups, so it is filtered as a whole.
            filtered.CreateAttribute("omniphysics:groupElemCounts0", Sdf.ValueTypeNames.UIntArray).Set(
                Vt.UIntArray([len(triangles)])
            )
            filtered.CreateAttribute("omniphysics:groupElemIndices0", Sdf.ValueTypeNames.UIntArray).Set(
                Vt.UIntArray(triangles)
            )


class ResetFromDataset(ManagerTermBase):
    """Reset the arm and the bag to rows of the physics-validated reset dataset.

    Rows are drawn by ``phase_weights``, so one training run sees home, the pregrasp, the open-jaw grasp and the closed
    grasp. Every row is a state a scripted attempt reached in physics, so none starts interpenetrating or off the strap.

    With ``sequential`` the term walks the rows of the single positive-weight phase in attempt order, which is how play
    and evaluation start every episode from the same home placements. Home rows can get a little bag position and
    heading noise, since at home the arm and the bag are not coupled.

    The phase of each environment's start is kept on ``env._hg_reset_phase``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.dataset = ResetDataset.load(params.get("dataset_path", DEFAULT_DATASET_PATH), env.device)
        validate_phase_weights(self.dataset, params.get("phase_weights", ALL_PHASES))
        self.robot = env.scene[ROBOT]
        self.bag = env.scene[BAG]
        joint_names = self.dataset.meta.get("joint_names")
        if joint_names is not None and list(joint_names) != list(self.robot.joint_names):
            raise ValueError(
                f"The reset dataset was generated for joints {list(joint_names)}, but the robot has"
                f" {list(self.robot.joint_names)}. Regenerate it."
            )
        stored, live = self.dataset.strap_nodes.shape[1], strap_rest_nodes(env).shape[0]
        if stored and stored != live:
            raise ValueError(
                f"The reset dataset stores {stored} strap nodes, but the scene's strap has {live}. Regenerate it."
            )
        self.cursor = 0
        env._hg_reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        dataset_path: str = str(DEFAULT_DATASET_PATH),
        phase_weights: dict[str, float] = ALL_PHASES,
        sequential: bool = False,
        home_position_noise: float = 0.0,
        home_heading_noise: float = 0.0,
    ) -> None:
        all_ids = torch.arange(env.num_envs, device=env.device)
        ids = all_ids if env_ids is None else all_ids[env_ids]
        n = ids.numel()

        if sequential:
            phase = validate_phase_weights(self.dataset, phase_weights)
            if len(phase) != 1:
                raise ValueError("Sequential resets need exactly one phase with a positive weight.")
            pool = self.dataset.rows(phase[0])
            rows = pool[(self.cursor + torch.arange(n, device=env.device)) % pool.numel()]
            self.cursor = (self.cursor + n) % pool.numel()
        else:
            rows = sample_rows(self.dataset, phase_weights, n)

        phase_id = self.dataset.phase[rows]
        joint_pos = self.dataset.joint_pos[rows]
        bag_pose = self.dataset.bag_pose[rows].clone()

        home = (phase_id == PHASES.index(HOME)).float()
        if home_position_noise > 0.0:
            bag_pose[:, :2] += home.unsqueeze(-1) * sample_uniform(
                -home_position_noise, home_position_noise, (n, 2), env.device
            )
        if home_heading_noise > 0.0:
            yaw = home * sample_uniform(-home_heading_noise, home_heading_noise, (n,), env.device)
            zeros = torch.zeros_like(yaw)
            bag_pose[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), bag_pose[:, 3:7])
        bag_pose[:, :3] += env.scene.env_origins[ids]

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=ids)
        # The drive holds the dataset pose until the first action, instead of pulling toward the last episode's target.
        self.robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=None, env_ids=ids)
        self.bag.write_root_pose_to_sim_index(root_pose=bag_pose, env_ids=ids)
        self.bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=ids)
        write_strap(env, ids, bag_pose, self.dataset.strap_nodes[rows])

        env._hg_reset_phase[ids] = phase_id


class randomize_encoder_bias(ManagerTermBase):
    """Sample a per-episode calibration bias [rad] for every joint encoder.

    The bias is added to the joint positions and targets the camera student observes
    (:func:`~.observations.encoder_joint_pos`), not to the physics, as a mis-calibrated real robot would report.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        env._hg_encoder_bias = torch.zeros(env.num_envs, env.scene[ROBOT].num_joints, device=env.device)

    def __call__(self, env: ManagerBasedEnv, env_ids: Sequence[int] | torch.Tensor | None, bound: float = 0.01) -> None:
        ids = slice(None) if env_ids is None else env_ids
        bias = env._hg_encoder_bias
        bias[ids] = sample_uniform(-bound, bound, bias[ids].shape, env.device)


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_name: str = "camera",
    position_range: float = 0.02,
    rotation_range: float = math.radians(3.0),
) -> None:
    """Move each environment's camera off its configured mount by up to ``position_range`` [m] per axis and tilt it
    by up to ``rotation_range`` [rad] about each of its own axes, as a hand-mounted real camera would be."""
    camera = env.scene.sensors[sensor_name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n = len(ids)
    nominal_pos = torch.tensor(camera.cfg.offset.pos, device=env.device)
    nominal_rot = torch.tensor(camera.cfg.offset.rot, device=env.device).expand(n, 4)
    pos = env.scene.env_origins[ids] + nominal_pos + sample_uniform(-position_range, position_range, (n, 3), env.device)
    angles = sample_uniform(-rotation_range, rotation_range, (n, 3), env.device)
    tilt = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    camera.set_world_poses(pos, quat_mul(nominal_rot, tilt), env_ids=ids, convention="ros")


def randomize_lights(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    dome_path: str = "/World/Ambient",
    key_path: str = "/World/OverheadLight",
    intensity_scale: tuple[float, float] = (0.5, 1.6),
    tint: float = 0.15,
    key_tilt: tuple[float, float] = (0.0, math.radians(55.0)),
) -> None:
    """Scale both lights' brightness, tint their colour, and point the key light from a random direction.

    The lights are shared by every environment, so this changes the whole scene at once; run it on an interval, not at
    reset, or it would flicker under episodes in progress. Each light's configured intensity is the nominal value.
    """
    from pxr import Gf

    rng = np.random.default_rng()
    stage = env.sim.stage
    for path in (dome_path, key_path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        intensity = prim.GetAttribute("inputs:intensity")
        nominal = prim.GetCustomDataByKey("hg_nominal_intensity")
        if nominal is None:
            nominal = float(intensity.Get())
            prim.SetCustomDataByKey("hg_nominal_intensity", nominal)
        intensity.Set(nominal * float(rng.uniform(*intensity_scale)))
        color = np.clip(1.0 + rng.uniform(-tint, tint, 3), 0.0, None)
        prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*(color / color.max())))
    key = stage.GetPrimAtPath(key_path)
    orient = key.GetAttribute("xformOp:orient") if key.IsValid() else None
    if orient is not None and orient.IsValid():
        # Tilt the light's -Z emission away from straight down, then turn it to a random heading.
        tilt, heading = float(rng.uniform(*key_tilt)), float(rng.uniform(-math.pi, math.pi))
        q = quat_mul(
            quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([heading])),
            quat_from_euler_xyz(torch.tensor([-tilt]), torch.zeros(1), torch.zeros(1)),
        )[0].tolist()
        quat_type = Gf.Quatd if isinstance(orient.Get(), Gf.Quatd) else Gf.Quatf
        orient.Set(quat_type(q[3], q[0], q[1], q[2]))
