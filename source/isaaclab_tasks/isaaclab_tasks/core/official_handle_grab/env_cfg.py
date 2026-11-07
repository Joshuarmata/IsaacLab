# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based handle-grab task, built the way the NVIDIA SO-101 vial task is.

The scene is the one :class:`~isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene.HandleGrabSceneCfg`
defines: an SO-101 over a tray, a rigid bag lying on its side, and a strap handle on the bag's top face. The task is
to close the jaws on the strap and keep the bag held, lifted 1 cm, for five seconds.

Compared with the direct task it replaces the demo-residual action with small relative joint steps, and the replayed
teacher with dense shaping, latched milestones, and resets from a physics-validated dataset. ``presets=sim2real`` adds
per-episode physical randomization; ``presets=pinch_only`` makes a held strap enough for success.
"""

import math

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import HandleGrabSceneCfg
from isaaclab_tasks.utils import preset

from . import mdp
from .mdp.task import ARM_JOINTS, EE_BODY_NAME, GRIPPER_JOINT, MOVING_JAW_BODY_NAME
from .mdp.terminations import HOLD_SECONDS, LIFT_HEIGHT
from .reset import ALL_PHASES, APPROACH_PHASES, CANONICAL_START, DEFAULT_DATASET_PATH

ARM_ACTION_SCALE = 0.033
"""Largest arm joint step per policy step [rad], the vial task's value."""
GRIPPER_ACTION_SCALE = 0.02
"""Largest jaw step per policy step [rad]. Also bounds the squeeze: a stalled jaw is commanded 0.02 rad past itself."""
LOWER_BODY_FINAL_WEIGHT = -4.0
LOWER_BODY_RAMP_STEPS = 16000
"""Environment steps over which the lower-body penalty grows to its full weight (about 250 PPO iterations)."""


@configclass
class ActionsCfg:
    """Joint targets relative to the measured position: a zero action holds still, and the drive cannot run away."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        scale=ARM_ACTION_SCALE,
        use_zero_offset=True,
    )
    gripper_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT],
        scale=GRIPPER_ACTION_SCALE,
        use_zero_offset=True,
    )


@configclass
class PolicyStateGroupCfg(ObsGroup):
    """Fully observed state actor inputs."""

    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot")})
    previous_action = ObsTerm(func=mdp.last_action)
    end_effector = ObsTerm(
        func=mdp.body_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME),
            "base_asset_cfg": SceneEntityCfg("robot"),
        },
    )
    bag = ObsTerm(
        func=mdp.body_state_b,
        params={"body_asset_cfg": SceneEntityCfg("bag"), "base_asset_cfg": SceneEntityCfg("robot")},
    )
    strap = ObsTerm(func=mdp.strap_in_base_frame)
    features = ObsTerm(func=mdp.grab_features)
    # Latched milestones make the once-per-episode milestone rewards Markov.
    progress = ObsTerm(func=mdp.progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    """Privileged training critic inputs."""

    contact = ObsTerm(func=mdp.jaw_contact_flags)


@configclass
class ObservationsCfg:
    policy: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


def _material_event(asset_cfg: SceneEntityCfg, static: tuple[float, float], dynamic: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": static,
            "dynamic_friction_range": dynamic,
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class EventsCfg:
    """Task-horizon resets from the physics-validated dataset plus the direct task's bag grip and servo variation."""

    bag_friction = _material_event(SceneEntityCfg("bag", body_names=".*"), (0.5, 1.2), (0.4, 1.0))
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.25),
            "damping_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_from_dataset = EventTerm(
        func=mdp.ResetFromDataset,
        mode="reset",
        params={
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "sequential": False,
            # pinch_only starts only before the grasp: a closed-grasp start is already a held strap.
            "phase_weights": preset(default=ALL_PHASES, pinch_only=APPROACH_PHASES),
            "home_position_noise": 0.005,
            "home_heading_noise": math.radians(3.0),
        },
    )


@configclass
class Sim2RealEventsCfg(EventsCfg):
    """Per-episode contact, motor and calibration variation for robot transfer; every episode starts at home."""

    jaw_material = _material_event(
        SceneEntityCfg("robot", body_names=[EE_BODY_NAME, MOVING_JAW_BODY_NAME]), (0.2, 1.3), (0.2, 1.3)
    )
    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_JOINT]),
            "stiffness_distribution_params": (0.6, 1.7),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.6, 2.5),
            "armature_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="reset", params={"bound": 0.01})

    def __post_init__(self):
        self.bag_friction.params.update(static_friction_range=(0.2, 1.3), dynamic_friction_range=(0.2, 1.3))
        self.arm_gains.params.update(stiffness_distribution_params=(0.85, 1.15), damping_distribution_params=(0.7, 1.5))
        self.reset_from_dataset.params.update(
            phase_weights=CANONICAL_START,
            home_position_noise=0.02,
            home_heading_noise=math.radians(15.0),
        )


@configclass
class RewardsCfg:
    """Dense approach and hold shaping, sparse milestones, a success bonus, two penalties, and light regularization."""

    approach_progress = RewTerm(func=mdp.approach_progress, weight=1.0)
    held_goal = RewTerm(func=mdp.held_goal, weight=0.1, params={"require_lift": preset(default=True, pinch_only=False)})
    # Grasp and lift, 1:2. pinch_only drops the lift milestone along with the lift requirement.
    milestones = RewTerm(
        func=mdp.milestones, weight=10.0, params={"weights": preset(default=(1.0, 2.0), pinch_only=(1.0, 0.0))}
    )
    success = RewTerm(func=mdp.success, weight=200.0)
    bag_lost = RewTerm(
        func=mdp.is_terminated_term, weight=-50.0, params={"term_keys": ["bag_out_of_bounds", "bag_dropped"]}
    )
    # Starts at zero and is ramped up by the curriculum: at full weight from step 0 the policy learned to avoid the bag.
    lower_body_contact = RewTerm(func=mdp.lower_body_contact, weight=0.0, params={"free_force": 1.0, "force_span": 5.0})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    # Joints 1-6 only: the gripper is meant to sit against its stop when it closes.
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)}
    )


@configclass
class TerminationsCfg:
    # Listed first: it updates the milestones that bag_dropped reads.
    success = DoneTerm(
        func=mdp.grab_success,
        params={
            "lift_height": LIFT_HEIGHT,
            "hold_s": HOLD_SECONDS,
            "require_lift": preset(default=True, pinch_only=False),
        },
    )
    bag_dropped = DoneTerm(func=mdp.bag_dropped)
    # Tray inner walls are x +-0.30 and y -0.12..0.30 about the environment origin, plus a margin.
    bag_out_of_bounds = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("bag"),
            "in_bound_range": {"x": (-0.34, 0.34), "y": (-0.16, 0.34), "z": (-0.05, 0.50)},
        },
    )
    unstable_robot = DoneTerm(func=mdp.abnormal_robot_state)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    lower_body_ramp = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.lower_body_contact.weight",
            "modify_fn": mdp.linear_weight_ramp,
            "modify_params": {"final_weight": LOWER_BODY_FINAL_WEIGHT, "ramp_steps": LOWER_BODY_RAMP_STEPS},
        },
    )


@configclass
class HandleGrabEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    scene: HandleGrabSceneCfg = HandleGrabSceneCfg(num_envs=1024, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = preset(default=EventsCfg(), sim2real=Sim2RealEventsCfg())
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Physics 60 Hz, policy 30 Hz. The arm's 0.5 rad/s speed limit moves the tip under a millimetre per physics
        # step, inside the contact skin; 120 Hz with 4 substeps did twice the solver work for the same policy rate.
        self.decimation = 2
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()

    def play_mode(self):
        """Play and evaluate complete episodes from the home pose, in dataset order.

        The nominal task pins the bag's friction and the servo gains; sim2real keeps its physical randomization.
        """
        super().play_mode()
        self.events.reset_from_dataset.params.update(sequential=True, phase_weights=CANONICAL_START)
        if not isinstance(self.events, Sim2RealEventsCfg):
            self.events.bag_friction.params.update(
                static_friction_range=(0.8, 0.8), dynamic_friction_range=(0.7, 0.7), restitution_range=(0.0, 0.0)
            )
            self.events.arm_gains.params.update(
                stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0)
            )
