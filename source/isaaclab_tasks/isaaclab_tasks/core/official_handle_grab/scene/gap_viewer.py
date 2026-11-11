# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the closed-mouth boxes that :func:`~..mdp.task.holding` tests, on the arm, in the viewer.

A row of the reset dataset (a grasp row by default) is written into the scene, the jaw is driven to the closed stop
(or held open), and the ``closed_mouth`` boxes from ``jaw_points.json`` are drawn on the rigid jaw every step: grey
while the jaw does not count as closed (nothing is registered), red when it is closed without the strap in a box,
green while :func:`holding` is true. A blue sphere marks the mouth's back point that the strap-depth reward
measures from. The grasp terms and the two placement rewards are printed once a second::

    uv run python -m isaaclab_tasks.core.official_handle_grab.scene.gap_viewer --viz kit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from ..mdp.task import (
    BAG,
    GRIPPER_CLOSED_POS,
    GRIPPER_HOLD_POS,
    GRIPPER_OPEN_POS,
    ROBOT,
    _strap_geometry,
    gripper_closed,
    gripper_joint_pos,
    holding,
    strap_grasp,
    write_strap,
)
from ..reset.curriculum import GRASP, PHASES
from ..reset.dataset import DEFAULT_DATASET_PATH, ResetDataset
from ..reset.generator import TASK, configure_env_cfg

JAW_TARGETS = {"closed": GRIPPER_CLOSED_POS, "open": GRIPPER_OPEN_POS}


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Draw the handle-grab closed-jaw gap box on the arm.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--phase", choices=PHASES, default=GRASP, help="Phase of the dataset row to show.")
    parser.add_argument("--row", type=int, default=0, help="Which row of that phase to show.")
    parser.add_argument(
        "--jaw", choices=(*JAW_TARGETS, "row"), default="closed", help="Jaw target; 'row' keeps the row's jaw angle."
    )
    parser.add_argument("--seconds", type=float, default=600.0, help="How long to keep the scene up.")
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    return setup_preset_cli(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args, hydra_args = _parse_args(argv)
    args.num_envs = 1

    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.utils import validate

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(TASK, None, overrides=hydra_args)
    configure_env_cfg(env_cfg, args)
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        try:
            with torch.inference_mode():
                _show(env, ResetDataset.load(args.dataset, device=env.device), args)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
    return 0


def _markers():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    def box(color):
        return sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.45)
        )

    def dot(color):
        return sim_utils.SphereCfg(radius=0.002, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

    gap = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/HandleGrab/closed_gap",
            markers={
                "holding": box((0.1, 0.9, 0.1)),
                "closed_no_strap": box((0.9, 0.1, 0.1)),
                "jaw_not_closed": box((0.6, 0.6, 0.6)),
            },
        )
    )
    back = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/HandleGrab/back_point", markers={"back": dot((0.1, 0.4, 1.0))})
    )
    return gap, back


def _box_pose_w(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World centers ``(boxes, 3)``, orientations ``(boxes, 4)`` and sizes ``(boxes, 3)`` of the closed-mouth boxes,
    and the world position ``(1, 3)`` of the mouth's back point."""
    from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul

    strap = _strap_geometry(env)
    gap = strap.mouth
    robot = env.scene[ROBOT]
    boxes = gap.bounds.shape[0]
    link_pos = robot.data.body_pos_w.torch[0, strap.rigid_body].expand(boxes, 3)
    link_quat = robot.data.body_quat_w.torch[0, strap.rigid_body].expand(boxes, 4)
    # Box coordinates are axes @ (p - origin), so the box frame's axes are the rows of ``axes`` in the link frame.
    center_link = gap.origin + gap.bounds.mean(dim=1) @ gap.axes
    box_in_link = quat_from_matrix(gap.axes.T.unsqueeze(0)).expand(boxes, 4)
    center = link_pos + quat_apply(link_quat, center_link)
    back = link_pos[:1] + quat_apply(link_quat[:1], (gap.origin + gap.back_point @ gap.axes).unsqueeze(0))
    return center, quat_mul(link_quat, box_in_link), gap.bounds[:, 1] - gap.bounds[:, 0], back


def _show(env, dataset: ResetDataset, args: argparse.Namespace) -> None:
    from ..env_cfg import ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE
    from ..reset.generator import Driver

    rows = torch.nonzero(dataset.phase == PHASES.index(args.phase)).flatten()
    if args.row >= rows.numel():
        raise SystemExit(f"[gap_viewer] {args.phase} has only {rows.numel()} rows.")
    row = int(rows[args.row])

    driver = Driver(env, ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE)
    robot, bag = env.scene[ROBOT], env.scene[BAG]
    ids = driver.ids
    env.reset()
    q = dataset.joint_pos[row : row + 1].clone()
    bag_pose = dataset.bag_pose[row : row + 1].clone()
    bag_pose[:, :3] += env.scene.env_origins
    robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
    robot.actuators.target_command.set_position_index(value=q, joint_ids=None, env_ids=ids)
    bag.write_root_pose_to_sim_index(root_pose=bag_pose, env_ids=ids)
    bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
    write_strap(env, ids, bag_pose, dataset.strap_nodes[row : row + 1])
    driver.trim.zero_()
    if args.jaw in JAW_TARGETS:
        q[:, 6] = JAW_TARGETS[args.jaw]

    strap = _strap_geometry(env)
    print(f"[gap_viewer] row={row} phase={args.phase} jaw={args.jaw}; closed counts below {GRIPPER_HOLD_POS:+.2f} rad")
    for lo, hi in (strap.mouth.bounds * 1e3).tolist():
        print(
            f"[gap_viewer]   box {-hi[1]:.0f}-{-lo[1]:.0f} mm from the hinge: {hi[0] - lo[0]:.1f} mm along the hinge"
            f" x {hi[2] - lo[2]:.1f} mm across"
        )
    gap_markers, back_marker = _markers()
    steps_per_print = max(1, round(1.0 / env.step_dt))
    for k in range(round(args.seconds / env.step_dt)):
        driver.step(q)
        held = holding(env)
        center, quat, size, back = _box_pose_w(env)
        kind = 0 if bool(held[0]) else (1 if bool(gripper_closed(env)[0]) else 2)
        kinds = torch.full((center.shape[0],), kind, device=env.device)
        gap_markers.visualize(center, quat, size, marker_indices=kinds)
        back_marker.visualize(back)
        if k % steps_per_print == 0:
            grasp = strap_grasp(env)
            along, back_dist = float(grasp.along[0]), float(grasp.back_dist[0])
            centered = 0.0 if along != along else 1.0 - abs(2.0 * along - 1.0)
            depth = 1.0 - math.tanh(back_dist / 0.02)
            print(
                f"[gap_viewer] t={k * env.step_dt:5.1f}s jaw={float(gripper_joint_pos(env)[0]):+.3f} rad "
                f"closed={bool(gripper_closed(env)[0])} side_to_side={bool(grasp.through[0])} "
                f"holding={bool(held[0])} along={along:.2f} back_dist={1e3 * back_dist:.1f} mm "
                f"centered={centered:.2f} depth={depth:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
