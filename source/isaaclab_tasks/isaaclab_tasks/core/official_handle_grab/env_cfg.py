# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based handle-grab task, built the way the NVIDIA SO-101 vial task is.

The scene is the one :class:`~isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene.HandleGrabSceneCfg`
defines: an SO-101 over a tray, a rigid bag lying on its side, and a strap handle on the bag's top face. The task is
grasp-only for now: close the jaws on the strap and keep it in them for five seconds. Lifting the bag is shelved, see
``LIFT_GRAVEYARD.md``.

Compared with the direct task it replaces the demo-residual action with small relative joint steps, and the replayed
teacher with dense shaping, latched milestones, and resets from a physics-validated dataset. ``presets=sim2real`` adds
per-episode physical randomization.
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

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import USE_DEFORMABLE_HANDLE, HandleGrabSceneCfg
from isaaclab_tasks.utils import preset

from . import mdp
from .mdp.task import ARM_JOINTS, EE_BODY_NAME, GRIPPER_JOINT, MOVING_JAW_BODY_NAME
from .mdp.terminations import HOLD_SECONDS
from .reset import ALL_PHASES, CANONICAL_START, DEFAULT_DATASET_PATH
from .reset.curriculum import GRASP_CLOSING

ARM_ACTION_SCALE = 0.033
"""Largest arm joint step per policy step [rad], the vial task's value."""
GRIPPER_ACTION_SCALE = 0.04
"""Largest jaw step per policy step [rad]. Also bounds the squeeze: a stalled jaw is commanded 0.04 rad past itself.

Twice the vial's 0.02: this jaw travels 0.9 rad from open to holding, which at 0.02 took 45 full-scale steps in a row
and was never found by exploration noise."""
LOWER_BODY_FINAL_WEIGHT = -4.0
LOWER_BODY_RAMP_STEPS = 16000
"""Environment steps over which the lower-body penalty grows to its full weight (about 250 PPO iterations)."""
CLOSING_FADE_STEPS = 19200
"""Environment steps over which ``grasp_closing`` starts fade from a quarter of the mix to none (300 PPO iterations)."""


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

    lower_body_force = ObsTerm(func=mdp.lower_body_force_feature)


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

    strap_attachment = EventTerm(func=mdp.attach_strap, mode="prestartup") if USE_DEFORMABLE_HANDLE else None
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
            "phase_weights": ALL_PHASES,
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
    """Dense approach, jaw-closing, hold and grasp-placement shaping, sparse milestones, a success bonus, two
    penalties, and light regularization."""

    approach_progress = RewTerm(func=mdp.approach_progress, weight=1.0)
    # Bridges reaching the strap and holding it: nothing else pays for the 0.9 rad close.
    jaw_closing = RewTerm(func=mdp.jaw_closing, weight=10.0, params={"std": 0.02})
    # The grasp itself (held_goal, milestones) out-pays every shaping and placement term; only the success bonus,
    # which is five seconds of the same grasp, is larger.
    held_goal = RewTerm(func=mdp.held_goal, weight=0.2)
    # While holding: a grasp near the top of the strap's arch, and a strap deep in the jaws, pay more.
    grasp_centered = RewTerm(func=mdp.grasp_centered, weight=0.1)
    strap_depth = RewTerm(func=mdp.strap_depth, weight=0.05, params={"std": 0.02})
    milestones = RewTerm(func=mdp.milestones, weight=20.0)
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
    success = DoneTerm(func=mdp.grab_success, params={"hold_s": HOLD_SECONDS})
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
    # The nearly closed starts teach the last part of the close; later in training every grasp starts further out.
    reset_fade = CurrTerm(
        func=mdp.fade_reset_phase,
        params={"phase": GRASP_CLOSING, "start_weight": 1.0, "final_weight": 0.0, "fade_steps": CLOSING_FADE_STEPS},
    )
    success_by_start = CurrTerm(func=mdp.success_by_start, params={"window": 200})


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
        # Physics 60 Hz, policy 30 Hz. At the commanded joint speeds (0.5 rad/s or less) the tip moves under a
        # millimetre per physics step, inside the contact skin; 120 Hz with 4 substeps did twice the solver work for
        # the same policy rate.
        self.decimation = 2
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # Deformable contacts overflow the default found/lost pair buffer.
        self.sim.physics = PhysxCfg(gpu_found_lost_pairs_capacity=2**22) if USE_DEFORMABLE_HANDLE else PhysxCfg()

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
