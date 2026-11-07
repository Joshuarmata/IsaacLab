# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Handle-grab scene: sandbox, SO-101, and bag.

Asset paths live in :mod:`.assets`, the vial example's ``assets/__init__.py``. The spawned
objects live on :class:`HandleGrabSceneCfg`, the vial example's ``SO101SceneCfg``: one robot,
one object, the support surface, a contact sensor, and the lights. The robot's ``init_state``
yaws the arm toward the bag; the wrapper USD stands the Y-up arm upright.
"""

from __future__ import annotations

import math

from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxCollisionCfg,
    PhysxConvexDecompositionCfg,
    PhysxDeformableBodyCfg,
    PhysxSurfaceDeformableBodyCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg
from isaaclab.utils import configclass

from .assets import ARM_USD, BAG_BODY_USD, BAG_USD, HANDLE_INFO, HANDLE_USD, JAW_POINTS, SANDBOX_USD

# Plain rigid tray. USE_SAND_PARTICLES stays False so the physics buffer stays the deformable-contact size.
USE_SAND_PARTICLES = False
SANDBOX_SPAWN_USD = str(SANDBOX_USD)
# Handle switch. True = the strap is a bendable PhysX surface deformable clamped to a body-only rigid bag (stiff enough
# to hold its arch under gravity, but yields when the jaws press it). False = the original fully rigid bag (body + strap
# as one rigid body, bag_single_body.usda). Needs the files made by assets/build_handle_assets.py.
USE_DEFORMABLE_HANDLE = False
BAG_SPAWN_USD = str(BAG_BODY_USD if USE_DEFORMABLE_HANDLE else BAG_USD)
HANDLE_SPAWN_USD = str(HANDLE_USD)
HANDLE_INFO_JSON = str(HANDLE_INFO)  # clamped vertices etc. (build_handle_assets.py)
JAW_POINTS_JSON = str(JAW_POINTS)  # proxy points on both jaws, for the geometric grasp test
ARM_SPAWN_USD = str(ARM_USD)

# ===========================================================================
# SANDBOX  (TUNE ME)
# The particle tray is a crop of sand.usd around x = 8.25. Those two numbers shift that crop onto the
# env origin and the floor onto z = 0. The plain tray (USE_SAND_PARTICLES = False) is already at the
# origin with its floor at z = 0, so the spawn pose is (0, 0, 0).
# ===========================================================================
SANDBOX_WORK_CENTER_X = 8.25
SANDBOX_FLOOR_Z = 0.4108
# Surface the bag rests on. 0 with the plain tray; ~3 cm when the particle sand bed is on.
SAND_SURFACE_Z = 0.030 if USE_SAND_PARTICLES else 0.0
# The bag still sits on TrayFloor (z = 0). The arm ignores that collider and hits an invisible
# cube TrayEeLimit whose top is this far below the visual floor, so the jaw can thread the loop.
EE_FLOOR_CLEARANCE = 0.04  # [m]

# ===========================================================================
# BAG  (from Rigid Bag.usd; bag frame = origin at the body centre)
#   body   : 0.165 x 0.089 x 0.152 m, z in [-0.076, 0.076]
#   handle : plate 0.065 x 0.013 x 0.038 m on top, z in [0.076, 0.114]
# The bag cannot stand on end (rounded bottom, it tips over in ~0.4 s) so it is spawned lying on a big
# face: 90 deg roll about x, then 180 deg yaw. Body centre is 0.0445 m above the tray. The extra yaw
# points the handle +y (into the open tray). Roll-only put the loop at y=-0.095, 10.5 cm in front of
# the arm and against the near wall — inside the SO-101's inner dead zone (usable reach is ~0.18-0.30 m
# in front of the base; the cube-stack task puts objects at 0.18-0.30 m).
# Handle after this rot is +0.095 m in y from the body centre. Hole axis is still world +Z.
# ===========================================================================
BAG_MASS = 0.08  # [kg] body + handle. Was 0.2 (the strap popped through the jaw under that load). Edit: your real bag.
BAG_HALF_THICKNESS = 0.0445
BAG_REST_Z = SAND_SURFACE_Z + BAG_HALF_THICKNESS + 0.005
BAG_REST_ROT = (0.0, 0.70710678, 0.70710678, 0.0)  # yaw 180 * roll 90, (x, y, z, w)
# Body at this xy keeps the loop ~0.265 m in front of ARM_BASE_POS y=-0.20, with clearance from the
# south wall (inner face y≈-0.12). Edit y toward 0 to push the handle farther (max ~0.30 m from the arm).
BAG_SPAWN_XY = (0.0, -0.03)
# Inner wall faces [m], from assets/sandbox_tray.usda. A cube of size 1 has half-extent = scale / 2,
# so the walls at x = ±0.305 and y = 0.305 / -0.125 (scale 0.01) leave this opening.
TRAY_INNER_X = (-0.300, 0.300)
TRAY_INNER_Y = (-0.120, 0.300)
# Footprint of the body once it is lying down: the 0.165 m side is x, the 0.152 m side is y.
BAG_FOOTPRINT_X = 0.165
BAG_FOOTPRINT_Y = 0.152
_BAG_WALL_MARGIN = 0.005  # [m] keep the body off the wall faces
# Half-diagonal of the lying body, plus a small gap. The center is kept this far inside every wall so a
# bag turned to any heading still clears the walls. The nominal spot is too close to the near wall for that,
# so spawns sit farther into the tray than (0, -0.03).
_BAG_REACH = math.hypot(0.5 * BAG_FOOTPRINT_X, 0.5 * BAG_FOOTPRINT_Y) + _BAG_WALL_MARGIN
BAG_X_RANGE = (
    TRAY_INNER_X[0] + _BAG_REACH - BAG_SPAWN_XY[0],
    TRAY_INNER_X[1] - _BAG_REACH - BAG_SPAWN_XY[0],
)
BAG_Y_RANGE = (
    TRAY_INNER_Y[0] + _BAG_REACH - BAG_SPAWN_XY[1],
    TRAY_INNER_Y[1] - _BAG_REACH - BAG_SPAWN_XY[1],
)
BAG_YAW_RANGE = (-math.pi, math.pi)  # full turn about world z, on top of the lying-flat rest rotation

# ===========================================================================
# D415  (overhead, one per env)
# Intel RealSense D415 color field of view is 69 x 42 deg. USD stores focal length and aperture in
# centimetres; the ratio is the field of view, and the 160 x 90 image keeps square pixels at 16:9.
# The strap is only a few pixels wide at this size. Raise CAMERA_WIDTH / CAMERA_HEIGHT_PX together
# (keep 16:9) if the pinch needs a sharper picture. Depth shares this color camera.
# ===========================================================================
D415_FOCAL_LENGTH_CM = 2.0
D415_HFOV_DEG = 69.0
D415_HORIZONTAL_APERTURE_CM = 2.0 * D415_FOCAL_LENGTH_CM * math.tan(math.radians(D415_HFOV_DEG / 2.0))
CAMERA_WIDTH = 160
CAMERA_HEIGHT_PX = 90
CAMERA_HEIGHT = 0.5  # [m] above the tray floor
# Center of the tray opening, half a metre up.
CAMERA_POS = (
    0.5 * (TRAY_INNER_X[0] + TRAY_INNER_X[1]),
    0.5 * (TRAY_INNER_Y[0] + TRAY_INNER_Y[1]),
    CAMERA_HEIGHT,
)
# ROS optical frame (+Z forward, +X right, +Y down). 180 deg about Y points +Z straight down.
CAMERA_ROT = (0.0, 1.0, 0.0, 0.0)
# The CAD base body. The student is told where this origin sits in the camera frame.
BASE_BODY_NAME = "base_08q"

# ===========================================================================
# ARM  (TUNE ME)
# SO_101_sim.usd is a 7-joint chain. Joint names are the prim names the URDF converter gave them:
#   revolute_1 base pan      revolute_2 shoulder lift   revolute_3 elbow       revolute_4 wrist pitch
#   revolute_5 wrist yaw     revolute_6 wrist roll      revolute_7 gripper jaw
# The CAD arm is Y-up. so101_physics.usda stands it up with a 90 deg roll about X on /Arm/Mount
# (CAD +Y becomes world +Z). Isaac Lab writes init_state.rot onto /Arm, the parent of Mount, so this
# quaternion is the 180 deg yaw that turns the upright arm toward the bag.
# ===========================================================================
TRAY_WALL_HEIGHT = 0.06  # [m] tray walls span z = 0 .. 0.06 (see assets/sandbox_tray.usda)
# The base sits on top of the wall height so the arm clears the south wall (y = -0.125) in front of it.
ARM_BASE_POS = (0.0, -0.20, TRAY_WALL_HEIGHT)
ARM_BASE_ROT = (0.0, 0.0, 1.0, 0.0)  # (x, y, z, w) 180 deg yaw
# Grip strength. The gripper motor used to be the same 3 N*m as every other joint: when the jaws stall on the strap
# the position error saturates it, so it squeezed with ~50 N (3 N*m / ~6 cm jaw lever) although ~2 N is enough to hold
# a light bag. Over-squeezing a 1 mm shell drives it into the jaw mesh, and the pull of the bag then snaps it
# through. The jaw and strap collision skins below stop the shell short of that mesh. Edit: raise if the bag
# slips out of the jaws; lower if the strap still pops through.
GRIPPER_EFFORT_LIMIT = 0.8  # [N*m] ~13 N total pinch
ARM_JOINT_NAMES = [f"revolute_{i}" for i in range(1, 8)]
ARM_GRIPPER_JOINT = "revolute_7"
# Home pose [rad]: every episode starts here (gripper revolute_7 starts OPEN at 0.6).
# Edit: change these to move the start/return pose. If the arm cannot reach the bag, raise the
# shoulder (revolute_2) or unfold the elbow (revolute_3). If it punches the floor, fold it up.
ARM_HOME_JOINT_POS = {
    "revolute_1": 0.11,
    "revolute_2": 0.94,
    "revolute_3": -2.17,
    "revolute_4": 0.71,
    "revolute_5": -0.09,
    "revolute_6": -0.37,
    "revolute_7": 0.6,
}
GRIPPER_OPEN_POS = 0.6  # [rad] jaw gap ~8 cm (what the episode starts with)
GRIPPER_CLOSED_POS = -0.38  # [rad] jaw gap ~1 cm = joint lower limit; holding the 1.3 cm handle plate stalls near -0.33
# The "hand" is the FIXED jaw on the URDF link wrist_roll_08c. That link's own visual is
# Wrist_Roll_08c.stl (see assembly_1.urdf). moving_jaw_08d is a *child rigid body* (revolute_7);
# walking every mesh under wrist_roll_08c used to pick the moving-jaw min-Y vertex
# (0.0216, -0.107, 0.0145), which sits 4.5 cm off the rigid-jaw tip and is why the policy
# stopped ~4 cm to the side of the loop.
# Tip = centroid of the distal 5 mm of Wrist_Roll_08c only, in the wrist_roll_08c body frame.
# JAW_INSERT_AXIS is that paddle's long axis (link -Y). The rotating jaw is penalised for
# hitting the bag; the rigid jaw is allowed to occupy the hole.
EE_BODY_NAME = "wrist_roll_08c"
EE_TIP_OFFSET = (-0.0065, -0.0934, 0.0464)
JAW_INSERT_AXIS = (0.0, -1.0, 0.0)
MOVING_JAW_BODY_NAME = "moving_jaw_08d"
# Rigid-jaw paddle length along gripper -Y, from Wrist_Roll_08c only: housing (y ≈ -0.040) to
# distal tip (y ≈ -0.095). Reach/insert target is this fraction PAST the loop centre.
RIGID_JAW_LENGTH = 0.055  # [m]
INSERT_THROUGH_FRAC = 0.90
INSERT_THROUGH = -INSERT_THROUGH_FRAC * RIGID_JAW_LENGTH  # -0.0495 m along bag +Y (through the hole)

# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------
SANDBOX_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Sandbox",
    spawn=sim_utils.UsdFileCfg(usd_path=SANDBOX_SPAWN_USD),
    # PhysX cannot replicate particle systems / PBD materials, so the particle tray is copied as plain USD.
    cloning_contexts=("isaaclab.cloner:UsdReplicateContext",) if USE_SAND_PARTICLES else None,
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(-SANDBOX_WORK_CENTER_X, 0.0, -SANDBOX_FLOOR_Z) if USE_SAND_PARTICLES else (0.0, 0.0, 0.0)
    ),
)

ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_SPAWN_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        # Tight contact generation so "EE vs bag" means real scrape, not the default ~2 cm PhysX halo.
        # The jaw links ship as one convexHull each. The hull of the wrist (housing plus the thin paddle)
        # is a solid block across the mouth, so the strap cannot sit between the jaws. A few convex pieces
        # follow the paddle. The deformable strap used an SDF of the graphics mesh instead; that query is
        # gone now that the strap is rigid.
        make_uninstanceable=True,
        collision_props={
            "/.*": [PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)],
            # Rest plane sits 1.5 mm outside the triangles, inside the ~1 cm closed mouth.
            "/.*(wrist_roll_08c|moving_jaw_08d).*": [PhysxCollisionCfg(contact_offset=0.006, rest_offset=0.0015)],
        },
        mesh_collision_props={
            # Few hulls: every extra piece is tested against the handle's pieces on every contact.
            "/.*(Wrist_Roll_08c|Moving_Jaw_08d).*": [
                PhysxConvexDecompositionCfg(
                    max_convex_hulls=8,
                    hull_vertex_limit=32,
                    voxel_resolution=200_000,
                    error_percentage=5.0,
                    min_thickness=0.001,
                    shrink_wrap=True,
                )
            ]
        },
        # fix_root_link pins the base to the world (Isaac Lab adds the anchor joint; see assets/so101_physics.usda)
        fix_root_link=True,
        # The scene solver runs this many iterations for every body, because it takes the max in the scene.
        # 16 was for the stiff deformable strap. 8 is enough for rigid-rigid contact and halves that work.
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=ARM_BASE_POS, rot=ARM_BASE_ROT, joint_pos=ARM_HOME_JOINT_POS),
    actuators={
        # stiffness 20 N*m/rad matches the 0.35 N*m/deg in your USD; max torque 3 N*m is the USD maxForce.
        # Edit: raise stiffness for snappier tracking (can overshoot / shake); raise effort_limit if
        # the arm cannot lift the bag. Split into two ImplicitActuatorCfg entries (arm vs
        # gripper) if you want different gains on revolute_7.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["revolute_[1-6]"],
            stiffness=20.0,
            damping=1.0,
            joint_effort_limit=3.0,
            # Solver clamp [rad/s]. Stock Isaac Lab SO-101 is 10. Also fills the actuator velocity
            # limit used by the PD model. Edit: raise toward 5-10 for snappier motion.
            joint_velocity_limit=0.5,
        ),
        # Same gains, but a weaker motor so the pinch cannot crush the strap (see GRIPPER_EFFORT_LIMIT).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[ARM_GRIPPER_JOINT],
            stiffness=20.0,
            damping=1.0,
            joint_effort_limit=GRIPPER_EFFORT_LIMIT,
            joint_velocity_limit=0.5,
        ),
    },
)

# Rigid-handle mode only: the strap is a thin ARCHED band (13 mm wide) and the USD ships it as a single convexHull
# collider, and the hull of an arch is a SOLID block: the opening does not exist for physics. Cook the handle as a
# convex decomposition instead (several convex pieces that follow the arch). The bag body keeps its convex hull.
# To inspect: Kit > Window > Debug > Physics > Collision > show collision shapes.
_RIGID_HANDLE_COLLISION = {
    "/Handle/.*": [
        PhysxConvexDecompositionCfg(
            # 8 pieces keep the arch open. More pieces multiply the jaw contacts and slow every step.
            max_convex_hulls=8,
            hull_vertex_limit=32,
            voxel_resolution=200_000,
            error_percentage=5.0,
            min_thickness=0.001,
            shrink_wrap=True,
        )
    ]
}

BAG_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bag",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BAG_SPAWN_USD,
        activate_contact_sensors=True,
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)]},
        mass_props={"": [sim_utils.MassCfg(mass=BAG_MASS)]},  # overrides physics:mass in the wrapper USD
        mesh_collision_props=None if USE_DEFORMABLE_HANDLE else _RIGID_HANDLE_COLLISION,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(*BAG_SPAWN_XY, BAG_REST_Z), rot=BAG_REST_ROT),
)

# ===========================================================================
# BAG HANDLE (deformable)  (TUNE ME; only used when USE_DEFORMABLE_HANDLE = True)
# A 13 x ~120 mm, 1 mm thick strap simulated as a PhysX surface deformable (cloth-like FEM shell). The two feet
# are clamped to the bag body (see HandleGrabEnv._setup_scene). It spawns at the bag's pose, so its mesh is in
# the bag frame.
#
# Feel of the strap (all stiffnesses in Pa, "force per area"):
#   HANDLE_STIFFNESS       one number that sets Young's modulus AND the stretch / shear / bend stiffness overrides
#                          (the schema's derived defaults turned out too soft: the arch collapsed at sim start).
#                          3e9 = rigid-ish plastic strap. Too floppy / sags: raise it (1e10, 3e10). Jaws cannot
#                          dent it: lower it (1e8..1e9). Very high values can make the solver jitter: raise
#                          HANDLE_SOLVER_ITERS then.
#   HANDLE_BEND_FACTOR     bend stiffness = HANDLE_STIFFNESS * this. Bending is what holds the arch up, so it is
#                          set higher than stretch.
#   HANDLE_THICKNESS       [m] shell thickness.
#   HANDLE_DAMPING         damping of stretch / bending; higher = settles faster, less ringing.
#   HANDLE_SOLVER_ITERS    position solver iterations. Stiff thin shells need many (cost grows with it).
# ===========================================================================
HANDLE_STIFFNESS = 3.0e9  # [Pa]
HANDLE_BEND_FACTOR = 10.0
HANDLE_THICKNESS = 0.001  # [m]
HANDLE_DENSITY = 1000.0  # [kg/m^3] -> the strap weighs ~1.5 g
HANDLE_DAMPING = 0.2
HANDLE_SOLVER_ITERS = 64

HANDLE_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Handle",
    # PhysX deformables are copied per env as plain USD (physics replication cannot clone them).
    cloning_contexts=("isaaclab.cloner:UsdReplicateContext",),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(*BAG_SPAWN_XY, BAG_REST_Z), rot=BAG_REST_ROT),
    spawn=sim_utils.UsdFileCfg(
        usd_path=HANDLE_SPAWN_USD,
        surface_deformable_props=[
            OmniPhysicsDeformableBodyCfg(kinematic_enabled=False),
            # A penetrated node is eased back out. A high depenetration speed is what snaps the strap
            # through the jaw once the shell has deformed past the mesh.
            PhysxDeformableBodyCfg(
                solver_position_iteration_count=HANDLE_SOLVER_ITERS,
                enable_speculative_c_c_d=True,
                max_depenetration_velocity=0.2,
            ),
            # Refresh jaw contacts every solver iteration and spend more of those iterations on collision.
            PhysxSurfaceDeformableBodyCfg(collision_pair_update_frequency=1, collision_iteration_multiplier=8.0),
        ],
        # Skin on the simulation mesh. rest_offset is the gap the shell keeps from the jaw; contact_offset
        # must be larger so the contact exists before the shell reaches that gap. Together with the jaw
        # skin (1.5 mm) each side rests 4 mm outside the mesh, which still fits the ~1 cm closed mouth.
        collision_props={"/sim_mesh": [PhysxCollisionCfg(rest_offset=0.0025, contact_offset=0.008)]},
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.1)),
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
            density=HANDLE_DENSITY,
            surface_thickness=HANDLE_THICKNESS,
            youngs_modulus=HANDLE_STIFFNESS,
            surface_stretch_stiffness=HANDLE_STIFFNESS,
            surface_shear_stiffness=HANDLE_STIFFNESS,
            surface_bend_stiffness=HANDLE_STIFFNESS * HANDLE_BEND_FACTOR,
            poissons_ratio=0.3,
            elasticity_damping=HANDLE_DAMPING,
            bend_damping=HANDLE_DAMPING,
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    ),
)

# Contact reporter on EVERY arm link, filtered to the bag. The env uses per-link forces and contact
# points to penalise collisions on the lower part of the bag body (not to detect the grasp).
ARM_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Arm/.*",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Bag"],
    history_length=1,
    # Once per policy step (sim dt 1/60, decimation 2). The reward reads contacts once per step, so a
    # download on every physics substep is wasted work. Keep this equal to decimation * dt.
    update_period=1.0 / 30.0,
    # Where each link touches the bag (average contact point). Collision uses bag-frame z of that point
    # so the top rim and handle are free; the bag is one rigid body so location is needed, not force alone.
    track_contact_points=True,
    max_contact_data_count_per_prim=16,
)

# Fill from the dome, plus a distant light shining straight down (-Z at the identity rotation).
AMBIENT_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/Ambient",
    spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(1.0, 1.0, 1.0)),
)
OVERHEAD_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/OverheadLight",
    spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), angle=2.0),
)

# D415 stereo depth is not valid inside about 5 cm. Readings closer than this are marked as no return.
D415_DEPTH_MIN_M = 0.05

# One D415 per env, fixed over the tray. RGB and depth update once per policy step.
D415_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/D415",
    offset=CameraCfg.OffsetCfg(pos=CAMERA_POS, rot=CAMERA_ROT, convention="ros"),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=D415_FOCAL_LENGTH_CM,
        horizontal_aperture=D415_HORIZONTAL_APERTURE_CM,
        focus_distance=CAMERA_HEIGHT,
        f_stop=0.0,
        clipping_range=(0.1, 2.0),
    ),
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT_PX,
    data_types=["rgb", "depth"],
    update_period=1.0 / 30.0,
)


@configclass
class HandleGrabSceneCfg(InteractiveSceneCfg):
    """Sandbox, SO-101, rigid bag, contact sensor, and lights.

    Same shape as the vial example's ``SO101SceneCfg``: the support surface, the robot (with its spawn
    pose on ``init_state``), the object, a contact sensor filtered to that object, and the lights. The
    camera is added by the camera task, as the vial adds its camera in ``camera_env_cfg``.
    """

    sandbox = SANDBOX_CFG
    robot = ARM_CFG
    bag = BAG_CFG
    contact: ContactSensorCfg = ARM_CONTACT_CFG
    ambient = AMBIENT_LIGHT_CFG
    overhead = OVERHEAD_LIGHT_CFG

    def __post_init__(self):
        super().__post_init__()
        if USE_DEFORMABLE_HANDLE:
            raise RuntimeError(
                "The manager-based handle-grab task senses the strap through contact points, which PhysX cannot"
                " report on deformables. Set USE_DEFORMABLE_HANDLE = False in handle_grab_scene.py."
            )
