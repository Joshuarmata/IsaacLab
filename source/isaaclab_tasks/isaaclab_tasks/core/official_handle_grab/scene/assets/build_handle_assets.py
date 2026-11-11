# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the deformable-handle assets listed by this package's ``__init__.py``.

Run once (and again whenever the source bag / arm USD files change):

    uv run python source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/scene/assets/build_handle_assets.py

Writes next to this script:
    bag_handle_ribbon.usda   the strap as a regular triangle grid (one Mesh under an Xform "Handle"), in the bag frame
    bag_handle_info.json     ribbon rest points, which vertices/triangles are clamped to the bag body, grid size
    jaw_points.json          sample points on the two gripper jaws (each in its own link frame), and the gap between
                             the jaws with the gripper at its closed stop (a box in the rigid-jaw link frame)

Why a new mesh: PhysX surface deformables simulate the triangle mesh as-is. The original strap has 780 vertices,
mostly in long thin slivers, which is a bad simulation mesh. The ribbon below follows the same centre line and
width (13 mm) but is a clean grid with 6.5 mm triangles (3 vertices across the strap).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

HERE = Path(__file__).parent
BAG_USD = "/home/joshua/Documents/Rigid Bag.usd"
STRAP_MESH = "/World/Part_Studio_1___Part_2/node_/mesh_"

STRAP_WIDTH = 0.013  # [m] along bag y (checked against the source mesh: y in [-0.0065, 0.0065])
# [m] target grid edge length. A coarse grid holds the arch in far fewer solver iterations: at 1024 envs, 6.5 mm with
# 32 iterations sags 2.4 mm for 13.6 ms per physics step; 2.2 mm with 64 iterations sagged 10 mm for 89.6 ms.
SPACING = 0.0065
LEG_EMBED = 0.004  # [m] the ribbon legs are extended this far into the bag body so they can be clamped there
ANCHOR_Z = 0.0795  # [m] ribbon vertices below this bag-frame height are clamped to the body (body face is z = 0.076)
JAW_TIP_Y = -0.045  # [m] jaw points kept: wrist_roll_08c-frame y below this (the whole mouth of both jaws)
JAW_SAMPLES = 128
GRIPPER_JOINT = "revolute_7"
# [m] Span of the closed mouth along the jaws, as distances from the hinge in the gap frame. The tips end ~85 mm out
# and the palm starts ~22 mm out (the gap shrinks to ~3 mm there); the mouth widens from ~8 mm at the tips to ~22 mm
# at the back.
MOUTH_FRONT = 0.065
MOUTH_BACK = 0.025
MOUTH_SLICES = 4
"""The closed mouth is a wedge, so it is a stack of boxes along the jaws, each as wide as the gap in its slice."""


def strap_centre_line() -> np.ndarray:
    """(n, 2) polyline (x, z) along the middle of the strap band, left foot to right foot."""
    stage = Usd.Stage.Open(BAG_USD)
    prim = stage.GetPrimAtPath(STRAP_MESH)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    pts = np.array([list(xf.Transform(Gf.Vec3d(*p))) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    xz = pts[:, [0, 2]]
    pivot = np.array([0.0, 0.07])
    angle = np.arctan2(xz[:, 1] - pivot[1], xz[:, 0] - pivot[0])
    xz = xz[np.argsort(angle)]
    # average consecutive groups of 6 points: each group straddles both faces of the 1 mm band
    groups = [xz[i : i + 6].mean(axis=0) for i in range(0, len(xz) - 5, 6)]
    line = np.array(groups)
    # symmetric about x = 0
    mirrored = line[::-1] * np.array([-1.0, 1.0])
    line = 0.5 * (line + mirrored)
    # the legs are vertical: extend both feet straight down into the body
    foot_l = np.array([line[0, 0], line[0, 1] - LEG_EMBED])
    foot_r = np.array([line[-1, 0], line[-1, 1] - LEG_EMBED])
    line = np.vstack((foot_l, line, foot_r))
    return line


def resample(line: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(2, int(round(s[-1] / spacing)) + 1)
    u = np.linspace(0.0, s[-1], n)
    return np.stack((np.interp(u, s, line[:, 0]), np.interp(u, s, line[:, 1])), axis=1)


def build_ribbon() -> dict:
    centre = resample(strap_centre_line(), SPACING)
    n_len = len(centre)
    n_w = max(1, int(round(STRAP_WIDTH / SPACING)))
    ys = np.linspace(-STRAP_WIDTH / 2, STRAP_WIDTH / 2, n_w + 1)
    points = np.zeros((n_len, n_w + 1, 3))
    points[:, :, 0] = centre[:, 0:1]
    points[:, :, 1] = ys[None, :]
    points[:, :, 2] = centre[:, 1:2]
    points = points.reshape(-1, 3)

    def vid(i, j):
        return i * (n_w + 1) + j

    tris = []
    for i in range(n_len - 1):
        for j in range(n_w):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris, dtype=np.int32)

    anchor_vertices = np.nonzero(points[:, 2] < ANCHOR_Z)[0]
    anchored = set(anchor_vertices.tolist())
    anchor_tris = [k for k, t in enumerate(tris) if any(v in anchored for v in t)]
    return {
        "points": points,
        "tris": tris,
        "anchor_vertices": anchor_vertices,
        "anchor_tris": np.array(anchor_tris, dtype=np.int32),
        "n_len": n_len,
        "n_w": n_w,
        "length": float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1))),
    }


def write_ribbon(r: dict) -> None:
    path = HERE / "bag_handle_ribbon.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Handle")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(
        "Strap of the bag handle as a clean triangle grid, in the bag frame. Generated by build_handle_assets.py. "
        "Becomes a PhysX surface deformable when spawned with surface_deformable_props."
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Handle/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(r["points"].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(r["tris"])))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(r["tris"].reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.1)])
    stage.GetRootLayer().Save()


def jaw_points() -> dict:
    """Mesh vertices of the pinching ends of both jaws, each in its own rigid-body (link) frame."""
    stage = Usd.Stage.Open(str(HERE / "so101_physics.usda"))
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    body_paths = {str(p.GetPath()) for p in bodies.values()}
    wrist = UsdGeom.Xformable(bodies["wrist_roll_08c"]).ComputeLocalToWorldTransform(0)
    to_wrist = wrist.GetInverse()

    def own_points(name: str) -> tuple[np.ndarray, np.ndarray]:
        """(points in the link frame, the same points in the wrist_roll_08c frame) for meshes owned by the link."""
        root = bodies[name]
        to_link = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).GetInverse()
        in_link, in_wrist = [], []
        it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != root and str(p.GetPath()) in body_paths:
                it.PruneChildren()
                continue
            if p.IsA(UsdGeom.Mesh):
                m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
                for v in UsdGeom.Mesh(p).GetPointsAttr().Get():
                    w = m.Transform(Gf.Vec3d(*v))
                    in_link.append(tuple(to_link.Transform(w)))
                    in_wrist.append(tuple(to_wrist.Transform(w)))
        return np.array(in_link), np.array(in_wrist)

    def farthest_samples(pts: np.ndarray, k: int) -> np.ndarray:
        chosen = [int(np.argmin(pts[:, 1]))]
        dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
        for _ in range(k - 1):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
        return pts[chosen]

    out = {}
    every = {}
    for key, name in (("rigid", "wrist_roll_08c"), ("moving", "moving_jaw_08d")):
        in_link, in_wrist = own_points(name)
        every[key] = in_link
        keep = in_wrist[:, 1] < JAW_TIP_Y
        print(f"[jaw_points] {name}: {len(in_link)} vertices, {int(keep.sum())} in the pinching end")
        out[key] = np.round(farthest_samples(in_link[keep], JAW_SAMPLES), 5).tolist()

    # Moving jaw at the gripper's closed stop: joint frame 1 is joint frame 0 turned about the joint's X axis.
    joint = next(UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.GetName() == GRIPPER_JOINT)
    if joint.GetAxisAttr().Get() != UsdPhysics.Tokens.x:
        raise RuntimeError(f"{GRIPPER_JOINT} turns about {joint.GetAxisAttr().Get()}, expected X.")
    body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    if stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0]).GetName() != "moving_jaw_08d":
        raise RuntimeError(f"{GRIPPER_JOINT} does not drive moving_jaw_08d.")

    def frame(pos, rot) -> Gf.Matrix4d:
        return Gf.Matrix4d(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(pos))

    closed_deg = joint.GetLowerLimitAttr().Get()
    joint0_w = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()) * (
        UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(0)
    )
    joint0_rigid = joint0_w * to_wrist
    joint1_inv = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()).GetInverse()

    def moving_at(deg: float, points=None) -> np.ndarray:
        turn = Gf.Matrix4d(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), deg), Gf.Vec3d(0.0))
        m = joint1_inv * turn * joint0_rigid
        return np.array([list(m.Transform(Gf.Vec3d(*p))) for p in (out["moving"] if points is None else points)])

    # The gap between the closed jaws, as a box in a hinge-aligned frame: axis 0 along the hinge, axis 2 the moving
    # jaw's closing direction (toward the rigid jaw), axis 1 along the jaws. Across the gap it spans from the moving
    # jaw's face to the rigid jaw's face; along the other two axes, where both jaws overlap.
    origin = np.array(joint0_rigid.Transform(Gf.Vec3d(0.0)))
    hinge = np.array(joint0_rigid.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
    hinge /= np.linalg.norm(hinge)
    moving_closed = moving_at(closed_deg)
    closing = (moving_at(closed_deg - 1.0) - moving_closed).mean(axis=0)
    closing -= hinge * closing.dot(hinge)
    closing /= np.linalg.norm(closing)
    axes = np.stack((hinge, np.cross(closing, hinge), closing))
    rigid_g = (np.array(out["rigid"]) - origin) @ axes.T
    moving_g = (moving_closed - origin) @ axes.T
    lo = np.maximum(rigid_g.min(axis=0), moving_g.min(axis=0))
    hi = np.minimum(rigid_g.max(axis=0), moving_g.max(axis=0))
    lo[2], hi[2] = moving_g[:, 2].max(), rigid_g[:, 2].min()
    if np.any(hi <= lo):
        raise RuntimeError(f"The closed jaws leave no gap between them: {lo} .. {hi}.")
    out["closed_gap"] = {
        "_comment": f"Gap between the jaws with {GRIPPER_JOINT} at its closed stop ({closed_deg:.1f} deg). A point p "
        "in the wrist_roll_08c (rigid-jaw) link frame is in the gap when axes @ (p - origin) lies within [min, max]. "
        "Axes (rows): hinge, along the jaws, closing direction (moving jaw toward rigid jaw) [m].",
        "origin": np.round(origin, 5).tolist(),
        "axes": np.round(axes, 6).tolist(),
        "min": np.round(lo, 5).tolist(),
        "max": np.round(hi, 5).tolist(),
    }
    size = (hi - lo) * 1000.0
    print(
        f"[jaw_points] closed-jaw gap {size[0]:.1f} mm along the hinge x {size[1]:.1f} mm along the jaws x "
        f"{size[2]:.1f} mm across"
    )

    # The whole closed mouth, from every jaw vertex rather than the tip samples. In each slice along the jaws the box
    # spans the hinge width both jaws share, and across from the moving jaw's face to the rigid jaw's face.
    rigid_all = (every["rigid"] - origin) @ axes.T
    moving_all = (moving_at(closed_deg, every["moving"]) - origin) @ axes.T
    boxes = []
    for front, back in zip(
        np.linspace(-MOUTH_FRONT, -MOUTH_BACK, MOUTH_SLICES + 1)[:-1],
        np.linspace(-MOUTH_FRONT, -MOUTH_BACK, MOUTH_SLICES + 1)[1:],
    ):
        r = rigid_all[(rigid_all[:, 1] >= front) & (rigid_all[:, 1] <= back)]
        m = moving_all[(moving_all[:, 1] >= front) & (moving_all[:, 1] <= back)]
        hinge_lo, hinge_hi = max(r[:, 0].min(), m[:, 0].min()), min(r[:, 0].max(), m[:, 0].max())
        r = r[(r[:, 0] >= hinge_lo) & (r[:, 0] <= hinge_hi)]
        m = m[(m[:, 0] >= hinge_lo) & (m[:, 0] <= hinge_hi)]
        moving_face = m[:, 2].max()
        rigid_face = r[:, 2][r[:, 2] > moving_face].min()
        lo, hi = np.array([hinge_lo, front, moving_face]), np.array([hinge_hi, back, rigid_face])
        if np.any(hi <= lo):
            raise RuntimeError(f"The closed mouth has no room at {front * 1e3:.0f} mm: {lo} .. {hi}.")
        boxes.append({"min": np.round(lo, 5).tolist(), "max": np.round(hi, 5).tolist()})
        size = (hi - lo) * 1000.0
        print(
            f"[jaw_points] mouth {-back * 1e3:.0f}-{-front * 1e3:.0f} mm from the hinge: {size[0]:.1f} mm along the "
            f"hinge x {size[2]:.1f} mm across"
        )
    out["closed_mouth"] = {
        "_comment": f"Space inside the jaws with {GRIPPER_JOINT} at its closed stop, as boxes in the closed_gap frame: "
        "a point p in the wrist_roll_08c link frame is in the mouth when axes @ (p - origin) lies within one box's "
        "[min, max] [m].",
        "origin": np.round(origin, 5).tolist(),
        "axes": np.round(axes, 6).tolist(),
        "boxes": boxes,
    }
    return out


def main() -> None:
    r = build_ribbon()
    write_ribbon(r)
    info = {
        "_comment": "Generated by build_handle_assets.py. Bag frame [m]; vertex order = bag_handle_ribbon.usda.",
        "n_len": r["n_len"],
        "n_w": r["n_w"],
        "rest_points": np.round(r["points"], 6).tolist(),
        "anchor_vertices": r["anchor_vertices"].tolist(),
        "anchor_triangles": r["anchor_tris"].tolist(),
    }
    (HERE / "bag_handle_info.json").write_text(json.dumps(info))
    (HERE / "jaw_points.json").write_text(json.dumps(jaw_points()))
    p = r["points"]
    print(
        f"[ribbon] {len(p)} vertices ({r['n_len']} x {r['n_w'] + 1}), {len(r['tris'])} triangles, "
        f"strap length {r['length'] * 1000:.1f} mm, width {STRAP_WIDTH * 1000:.0f} mm"
    )
    print(f"[ribbon] bbox min {p.min(0).round(4)} max {p.max(0).round(4)}")
    print(f"[ribbon] clamped to the body: {len(r['anchor_vertices'])} vertices, {len(r['anchor_tris'])} triangles")


if __name__ == "__main__":
    main()
