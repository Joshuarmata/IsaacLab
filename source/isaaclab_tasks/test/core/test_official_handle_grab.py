# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests of the manager-based handle-grab task: milestones, rewards, reset dataset, geometry, presets, eval."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.core.official_handle_grab import evaluation
from isaaclab_tasks.core.official_handle_grab.mdp.curriculums import (
    fade_reset_phase,
    linear_weight_ramp,
    success_by_start,
)
from isaaclab_tasks.core.official_handle_grab.mdp.progress import GrabProgress
from isaaclab_tasks.core.official_handle_grab.mdp.task import (
    EE_TIP_OFFSET,
    JAW_INSERT_AXIS,
    JawGap,
    load_grasp_goal,
    strap_in_mouth,
    strap_through_gap,
)
from isaaclab_tasks.core.official_handle_grab.reset import generator, viewer
from isaaclab_tasks.core.official_handle_grab.reset.curriculum import ALL_PHASES, GRASP, GRASP_CLOSING, HOME, PHASES
from isaaclab_tasks.core.official_handle_grab.reset.dataset import ResetDataset, sample_rows, validate_phase_weights
from isaaclab_tasks.core.official_handle_grab.scene import handle_grab_scene as scene
from isaaclab_tasks.utils.hydra import resolve_task_config

# The package's lazy ``mdp`` namespace re-exports Isaac Lab's own ``rewards`` module, so import ours by its full path.
rewards = importlib.import_module("isaaclab_tasks.core.official_handle_grab.mdp.rewards")

##
# Milestones.
##


def _run(progress: GrabProgress, holding: list[list[bool]], start: int = 0) -> None:
    """Feed per-step ``holding`` values (one inner list per step, one entry per environment)."""
    for step, hold in enumerate(holding, start=start):
        progress.update(torch.tensor(hold), torch.full((len(hold),), step))


def test_success_needs_the_strap_held_for_the_whole_hold_time():
    progress = GrabProgress(2, "cpu", hold_steps=5, grasp_steps=2)
    # Env 0 holds the strap from step 0; env 1 never does.
    _run(progress, [[True, False]] * 6)
    assert progress.grasped.tolist() == [True, False]
    assert progress.success.tolist() == [True, False]
    assert progress.time_to_success.tolist() == [4, -1]  # five consecutive held steps: 0..4


def test_losing_the_strap_restarts_the_hold_and_only_a_sustained_loss_is_a_drop():
    progress = GrabProgress(1, "cpu", hold_steps=4, grasp_steps=1, drop_steps=3)
    _run(progress, [[True], [True], [True], [False]])
    assert progress.grasped[0] and not progress.dropped[0]  # one lost step is a flicker, not a drop
    _run(progress, [[True]] * 3, start=4)
    # The hold restarted at the flicker: 3 live steps since it, one short of the 4 needed.
    assert not progress.success[0] and not progress.dropped[0]
    _run(progress, [[False]] * 3, start=7)
    assert progress.dropped[0] and not progress.success[0]


def test_reset_clears_only_the_requested_environments():
    progress = GrabProgress(2, "cpu", hold_steps=1, grasp_steps=1)
    _run(progress, [[True, True]] * 2)
    progress.reset([1])
    assert progress.success.tolist() == [True, False]
    assert progress.time_to_success.tolist() == [0, -1]


##
# Rewards and curriculum.
##


def test_strap_counts_in_the_mouth_when_a_segment_crosses_any_box_not_the_space_between():
    # Two boxes along y in the link frame, a 10 mm empty band between them, like the slices of the closed mouth.
    mouth = JawGap.from_json(
        {
            "origin": [0.0, 0.0, 0.0],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "boxes": [
                {"min": [-0.01, -0.03, -0.01], "max": [0.01, -0.02, 0.01]},
                {"min": [-0.01, 0.02, -0.01], "max": [0.01, 0.03, 0.01]},
            ],
        },
        "cpu",
    )
    # One-column straps crossing the jaws along x: through the second box, and through the empty band.
    line = torch.tensor([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    nodes = torch.stack((line + torch.tensor([0.0, 0.025, 0.0]), line))
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    assert strap_through_gap(nodes, torch.zeros(2, 3), identity, mouth, columns=1).tolist() == [True, False]
    # The single closed_gap box (the direct task's entry) loads as a one-box stack.
    box = {"origin": [0.0] * 3, "axes": mouth.axes.tolist(), "min": [-0.01] * 3, "max": [0.01] * 3}
    single = JawGap.from_json(box, "cpu")
    assert single.bounds.shape == (1, 2, 3)
    assert strap_through_gap(nodes, torch.zeros(2, 3), identity, single, columns=1).tolist() == [False, True]


def test_grasp_needs_the_strap_to_cross_the_mouth_side_to_side_not_through_the_front_or_along_the_jaws():
    # One box in the link frame: axis 0 is the hinge (the gripper's sides), axis 1 along the jaws, axis 2 across them.
    mouth = JawGap.from_json(
        {
            "origin": [0.0, 0.0, 0.0],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "boxes": [{"min": [-0.01, -0.03, -0.005], "max": [0.01, -0.02, 0.005]}],
        },
        "cpu",
    )
    x = torch.linspace(-0.035, 0.035, 8)
    side_to_side = torch.stack((x, torch.full_like(x, -0.025), torch.zeros_like(x)), dim=-1)
    # Enters through a side face, then leaves through the front face (axis 1 below -0.03) before the other side.
    out_the_front = torch.tensor(
        [
            [-0.035, -0.025, 0],
            [-0.025, -0.025, 0],
            [-0.015, -0.025, 0],
            [-0.005, -0.025, 0],
            [0, -0.035, 0],
            [0, -0.045, 0],
            [0.015, -0.045, 0],
            [0.025, -0.045, 0],
        ]
    )
    y = torch.linspace(-0.045, -0.005, 8)
    along_the_jaws = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), dim=-1)
    nodes = torch.stack((side_to_side, out_the_front, along_the_jaws))
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3)
    grasp = strap_in_mouth(nodes, torch.zeros(3, 3), identity, mouth, columns=1)
    assert grasp.through.tolist() == [True, False, False]
    assert float(grasp.along[0]) == pytest.approx(0.5)  # nodes 2 and 5 bracket the crossing, the middle of 8
    assert torch.isnan(grasp.along[1:]).all()
    # The back point is the centre of the box's back face, (0, -0.02, 0); the first strap passes 5 mm in front of it.
    assert float(grasp.back_dist[0]) == pytest.approx(0.005, abs=1e-6)
    # The direct task's looser test counts any strap touching the box, including the two rejected here.
    assert strap_through_gap(nodes, torch.zeros(3, 3), identity, mouth, columns=1).tolist() == [True, True, True]


def _reward_env(progress: GrabProgress) -> SimpleNamespace:
    return SimpleNamespace(num_envs=progress.grasped.numel(), device="cpu", _hg_progress=progress)


def test_approach_reward_pays_progress_only_before_the_grasp_and_not_on_the_first_step(monkeypatch):
    progress = GrabProgress(1, "cpu")
    env = _reward_env(progress)
    term = rewards.approach_progress(SimpleNamespace(params={}), env)
    distances = iter([0.20, 0.195, 0.19, 0.05])
    monkeypatch.setattr(rewards, "tip_pos_w", lambda _: torch.tensor([[next(distances), 0.0, 0.0]]))
    monkeypatch.setattr(rewards, "strap_target_w", lambda _: torch.zeros(1, 3))

    paid = [float(term(env, scale=0.01)) for _ in range(3)]
    assert paid[0] == 0.0  # no previous distance after a reset
    assert paid[1:] == pytest.approx([0.5, 0.5], abs=1e-4)
    progress.grasped[:] = True
    assert float(term(env, scale=0.01)) == 0.0  # a closing hand is no longer approaching


def test_jaw_closing_pays_closing_at_the_strap_once_and_not_reopening_and_closing_again(monkeypatch):
    env = _reward_env(GrabProgress(1, "cpu"))
    term = rewards.jaw_closing(SimpleNamespace(params={}), env)
    # (tip distance to the strap [m], closed fraction) per step.
    states = iter([(0.0, 0.0), (0.0, 0.5), (0.0, 0.0), (0.0, 0.5), (0.3, 1.0), (0.0, 1.0)])
    state = {}

    def tip(_):
        state["d"], state["f"] = next(states)
        return torch.tensor([[state["d"], 0.0, 0.0]])

    monkeypatch.setattr(rewards, "tip_pos_w", tip)
    monkeypatch.setattr(rewards, "strap_target_w", lambda _: torch.zeros(1, 3))
    monkeypatch.setattr(rewards, "gripper_closed_fraction", lambda _: torch.tensor([state["f"]]))

    paid = [float(term(env, std=0.02)) for _ in range(6)]
    # Baseline, half closed at the strap, reopened, half closed again, fully closed far away, fully closed at the strap.
    assert paid == pytest.approx([0.0, 0.5, 0.0, 0.0, 0.0, 0.5], abs=1e-4)


def test_grasp_milestone_pays_once_per_episode_and_not_on_the_first_step():
    progress = GrabProgress(1, "cpu")
    env = _reward_env(progress)
    term = rewards.milestones(SimpleNamespace(params={}), env)
    progress.grasped[:] = True
    assert float(term(env)[0]) == 0.0  # already latched on the first step after the reset
    assert float(term(env)[0]) == 0.0
    term.reset(None)
    progress.grasped[:] = False
    term(env)
    progress.grasped[:] = True
    assert float(term(env)[0]) == 1.0
    assert float(term(env)[0]) == 0.0


def test_lower_body_penalty_ramp_grows_linearly_and_saturates():
    ramp = linear_weight_ramp
    env = SimpleNamespace(common_step_counter=0)
    kwargs = dict(current_weight=0.0, final_weight=-4.0, ramp_steps=1000, start_steps=200)
    assert ramp(env, None, **kwargs) == 0.0
    env.common_step_counter = 700
    assert ramp(env, None, **kwargs) == pytest.approx(-2.0)
    env.common_step_counter = 10**6
    assert ramp(env, None, **kwargs) == -4.0


def test_reset_fade_moves_one_phase_weight_and_leaves_a_mix_without_it_alone():
    term = SimpleNamespace(params={"phase_weights": dict(ALL_PHASES)})
    env = SimpleNamespace(common_step_counter=0, event_manager=SimpleNamespace(get_term_cfg=lambda _: term))
    kwargs = dict(phase=GRASP_CLOSING, start_weight=1.0, final_weight=0.0, fade_steps=1000)
    assert fade_reset_phase(env, None, **kwargs) == 1.0
    env.common_step_counter = 500
    assert fade_reset_phase(env, None, **kwargs) == pytest.approx(0.5)
    env.common_step_counter = 10**6
    assert fade_reset_phase(env, None, **kwargs) == 0.0
    assert term.params["phase_weights"] == {**ALL_PHASES, GRASP_CLOSING: 0.0}
    assert ALL_PHASES[GRASP_CLOSING] == 1.0  # the shared default mix is not modified
    term.params["phase_weights"] = {HOME: 1.0}
    assert fade_reset_phase(env, None, **kwargs) is None
    assert term.params["phase_weights"] == {HOME: 1.0}


def test_success_by_start_splits_finished_episodes_by_their_reset_phase():
    progress = GrabProgress(4, "cpu")
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        _hg_progress=progress,
        _hg_reset_phase=torch.tensor([PHASES.index(HOME), PHASES.index(HOME), PHASES.index(GRASP_CLOSING), 0]),
        episode_length_buf=torch.tensor([10, 10, 10, 0]),
    )
    term = success_by_start(SimpleNamespace(params={}), env)
    progress.success[:] = torch.tensor([True, False, True, True])
    rates = term(env, slice(None), window=200)
    # Env 3 never stepped (the reset before the first step), so it is not an episode.
    assert rates[HOME] == pytest.approx(0.5)
    assert rates[GRASP_CLOSING] == pytest.approx(1.0)
    assert rates[GRASP] == 0.0
    progress.success[:] = False
    rates = term(env, torch.tensor([0, 1]), window=200)
    assert rates[HOME] == pytest.approx(0.25)  # four home episodes, one success
    assert rates[GRASP_CLOSING] == pytest.approx(1.0)  # not among the envs reset this time


##
# Reset dataset.
##


def _dataset(attempts: int = 5) -> ResetDataset:
    rows = attempts * len(PHASES)
    phase = torch.arange(len(PHASES)).repeat_interleave(attempts)
    attempt = torch.arange(attempts).repeat(len(PHASES))
    joint_pos = torch.arange(rows, dtype=torch.float32).unsqueeze(-1).repeat(1, 7)
    bag_pose = torch.zeros(rows, 7)
    bag_pose[:, 6] = 1.0
    return ResetDataset(phase, attempt, joint_pos, bag_pose, {"joint_names": [f"revolute_{i}" for i in range(1, 8)]})


def test_dataset_round_trips_through_weights_only_load_and_keeps_attempt_order(tmp_path):
    dataset = _dataset()
    dataset.attempt[:5] = torch.tensor([3, 1, 4, 0, 2])  # shuffled home rows
    dataset.save(tmp_path / "reset.pt")
    loaded = ResetDataset.load(tmp_path / "reset.pt")
    assert loaded.meta["joint_names"][-1] == "revolute_7"
    assert loaded.attempt[loaded.rows(HOME)].tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(loaded.joint_pos, dataset.joint_pos)


def test_viewer_plays_one_attempt_through_its_phases_before_the_next_attempt():
    phase = torch.tensor([2, 0, 1, 0, 1, 2])
    attempt = torch.tensor([0, 1, 0, 0, 1, 1])
    dataset = ResetDataset(phase, attempt, torch.zeros(6, 7), torch.zeros(6, 7))
    order = viewer.row_order(dataset)
    assert [(int(attempt[i]), int(phase[i])) for i in order] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_phase_sampling_draws_only_weighted_phases_in_proportion():
    dataset = _dataset()
    torch.manual_seed(0)
    rows = sample_rows(dataset, {HOME: 1.0, GRASP: 3.0}, 4000)
    phases = dataset.phase[rows]
    assert set(phases.tolist()) == {PHASES.index(HOME), PHASES.index(GRASP)}
    assert (phases == PHASES.index(GRASP)).float().mean() == pytest.approx(0.75, abs=0.05)


@pytest.mark.parametrize(
    "weights",
    [{"reorient": 1.0}, {HOME: -1.0}, {HOME: 0.0}],
    ids=["unknown_phase", "negative_weight", "no_positive_weight"],
)
def test_phase_weights_are_validated_against_the_dataset(weights):
    with pytest.raises(ValueError):
        validate_phase_weights(_dataset(), weights)


def test_missing_phase_rows_are_reported():
    dataset = _dataset()
    keep = dataset.phase != PHASES.index(GRASP)
    partial = ResetDataset(dataset.phase[keep], dataset.attempt[keep], dataset.joint_pos[keep], dataset.bag_pose[keep])
    with pytest.raises(ValueError, match=GRASP):
        validate_phase_weights(partial, ALL_PHASES)


##
# Generator geometry.
##


def test_wrist_targets_put_the_tip_on_the_recorded_pinch_for_any_bag_pose():
    goal = load_grasp_goal()
    assert goal.jaw_quat_bag is not None
    torch.manual_seed(0)
    poses = generator.sample_bag_poses(64, 0.03, 0.04, math.radians(20))
    pos, quat = poses[:, :3], poses[:, 3:]
    grasp, pregrasp, wrist_quat = generator.wrist_targets(pos, quat, goal.tip_pos_bag, goal.jaw_quat_bag, 0.06)

    tip = grasp + quat_apply(wrist_quat, torch.tensor(EE_TIP_OFFSET).expand_as(grasp))
    pinch = pos + quat_apply(quat, torch.tensor(goal.tip_pos_bag).expand_as(pos))
    assert torch.allclose(tip, pinch, atol=1e-5)
    insert = quat_apply(wrist_quat, torch.tensor(JAW_INSERT_AXIS).expand_as(grasp))
    assert torch.allclose(grasp - pregrasp, 0.06 * insert, atol=1e-5)  # backed off against the insertion direction


def test_sampled_bag_poses_stay_in_the_spawn_box_and_clear_of_the_tray_walls():
    torch.manual_seed(0)
    x_half, y_half, yaw_half = 0.03, 0.04, math.radians(20)
    poses = generator.sample_bag_poses(2000, x_half, y_half, yaw_half)
    spawn = torch.tensor(scene.BAG_SPAWN_XY)
    assert ((poses[:, :2] - spawn).abs() <= torch.tensor([x_half, y_half]) + 1e-6).all()
    assert torch.allclose(poses[:, 3:].norm(dim=-1), torch.ones(2000), atol=1e-5)
    # The nominal spot is 14 mm from the near wall, so the box's -y half must have been cut back to the footprint.
    assert poses[:, 1].min() > scene.TRAY_INNER_Y[0] + 0.5 * scene.BAG_FOOTPRINT_Y


##
# Inverse kinematics.
##

_LINKS = (0.30, 0.25, 0.10)
_POS_W, _ROT_W = generator.IK_POSITION_WEIGHT, generator.IK_ROTATION_WEIGHT


def _planar_arm(target_q: torch.Tensor):
    """A 3-link planar arm as an ``evaluate`` callback: weighted pose residual (6 rows) and Jacobian (6 x 3)."""
    link = torch.tensor(_LINKS)

    def pose(q):
        angle = q.cumsum(dim=-1)
        return (link * angle.cos()).sum(-1), (link * angle.sin()).sum(-1), angle[..., -1]

    gx, gy, gyaw = pose(target_q)

    def evaluate(q):
        x, y, yaw = pose(q)
        angle = q.cumsum(dim=-1)
        # Joint k moves every link from k outward, so the tip's partials are suffix sums over the links.
        dx = -(link * angle.sin()).flip(-1).cumsum(-1).flip(-1)
        dy = (link * angle.cos()).flip(-1).cumsum(-1).flip(-1)
        r = torch.zeros(q.shape[0], 6)
        r[:, 0], r[:, 1], r[:, 5] = (gx - x) * _POS_W, (gy - y) * _POS_W, (gyaw - yaw) * _ROT_W
        jac = torch.zeros(q.shape[0], 6, 3)
        jac[:, 0], jac[:, 1], jac[:, 5] = dx * _POS_W, dy * _POS_W, _ROT_W
        return r, jac

    return evaluate


def _lm(evaluate, q0, low=-3.0, high=3.0, iterations=80):
    low, high = torch.full_like(q0, low), torch.full_like(q0, high)
    return generator.levenberg_marquardt(evaluate, q0, low, high, iterations=iterations, position_weight=_POS_W)


def test_levenberg_marquardt_reaches_a_reachable_pose_from_a_start_far_from_it():
    torch.manual_seed(0)
    truth = torch.tensor([[0.4, 0.5, -0.3]]).repeat(16, 1)
    q, r, history = _lm(_planar_arm(truth), truth + 0.8 * torch.randn(16, 3))
    assert (r[:, :3].norm(dim=-1) / _POS_W).max() < 1e-4  # position error [m]
    assert (r[:, 5].abs() / _ROT_W).max() < 1e-3  # yaw error [rad]
    assert history[0] > 10 * history[-1]


def test_levenberg_marquardt_never_raises_the_cost_and_stays_inside_the_limits():
    torch.manual_seed(1)
    truth = torch.tensor([[0.9, -0.4, 0.6]])
    evaluate, start = _planar_arm(truth), torch.zeros(1, 3)
    costs = [float((_lm(evaluate, start, -1.0, 1.0, iterations=k)[1] ** 2).sum()) for k in range(1, 12)]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(costs, costs[1:]))
    q, _, _ = _lm(_planar_arm(torch.tensor([[2.5, 2.5, 2.5]])), start, -1.0, 1.0)  # target beyond the limits
    assert q.min() >= -1.0 and q.max() <= 1.0


def test_levenberg_marquardt_stops_and_stays_finite_on_an_unreachable_pose():
    far = torch.tensor([[0.0, 0.0, 0.0]])
    evaluate = _planar_arm(far)

    def too_far(q):
        r, jac = evaluate(q)
        r[:, 0] += 5.0 * _POS_W  # a target 5 m beyond the arm's reach
        return r, jac

    q, r, history = _lm(too_far, torch.tensor([[0.3, 0.2, 0.1]]), iterations=200)
    assert torch.isfinite(q).all() and torch.isfinite(r).all()
    assert len(history) < 200  # the damping grew until nothing more could be gained
    assert history[-1] <= history[0]


def _candidates(q, pos, rot):
    return torch.tensor(q, dtype=torch.float32), torch.tensor(pos), torch.tensor(rot)


def test_ik_selection_keeps_the_valid_start_nearest_the_seed_and_rejects_limit_pinned_ones():
    seed = torch.zeros(1, 2)
    low, high = torch.full((2,), -1.0), torch.full((2,), 1.0)
    q, pos, rot = _candidates(
        # far valid, nearest valid, nearer but too close to a joint limit, nearest but misses the position
        [[[0.8, 0.0], [0.3, 0.1], [0.99, 0.0], [0.05, 0.0]]],
        [[0.001, 0.002, 0.001, 0.05]],
        [[0.1, 0.1, 0.1, 0.1]],
    )
    pick = generator.select_ik_solution(q, pos, rot, seed, low, high, pos_tol=0.005, rot_tol=0.31, margin=0.03)
    assert pick.valid.tolist() == [True] and pick.index.tolist() == [1] and pick.valid_count.tolist() == [2]
    assert pick.delta[0] == pytest.approx(math.hypot(0.3, 0.1))


def test_ik_selection_accepts_a_start_clamped_onto_the_inset_search_limit():
    seed = torch.zeros(1, 1)
    low, high = torch.tensor([-2.2275]), torch.tensor([0.9141])  # revolute_4 limits [rad], float32
    q = (high - 0.03).reshape(1, 1, 1)  # where WristIK.solve clamps a start pushed against the stop
    pick = generator.select_ik_solution(
        q, torch.tensor([[0.001]]), torch.tensor([[0.1]]), seed, low, high, pos_tol=0.005, rot_tol=0.31, margin=0.03
    )
    assert pick.valid.tolist() == [True]


def test_ik_selection_reports_the_best_position_error_when_no_start_is_valid_and_ignores_nan():
    seed = torch.zeros(2, 2)
    low, high = torch.full((2,), -1.0), torch.full((2,), 1.0)
    nan = float("nan")
    q, pos, rot = _candidates(
        [[[0.1, 0.0], [0.2, 0.0]], [[nan, 0.0], [0.4, 0.0]]],
        [[0.04, 0.02], [0.0, 0.5]],
        [[0.1, 0.1], [0.1, 0.1]],
    )
    pick = generator.select_ik_solution(q, pos, rot, seed, low, high, pos_tol=0.005, rot_tol=0.31, margin=0.03)
    assert pick.valid.tolist() == [False, False]
    assert pick.index.tolist() == [1, 0]  # smallest position error, even though row 1's NaN start reads 0.0 m
    assert pick.valid_count.tolist() == [0, 0]


def test_seed_bank_keeps_one_child_per_parent_and_retries_the_ones_without_one():
    bank = generator._Bank()
    joints = torch.zeros(3, 7)
    pose = torch.zeros(3, 7)
    assert bank.add(joints, pose, torch.tensor([4, 4, 9])) == 2
    child = generator._Bank()
    assert child.add(joints[:1], pose[:1], torch.tensor([4])) == 1
    drawn = set()
    for _ in range(30):
        _joints, _pose, attempt = bank.sample(4, "cpu", child)
        drawn.update(int(v) for v in attempt.tolist())
    assert drawn == {9}


def test_randomized_ik_selection_draws_only_valid_starts_and_covers_all_of_them():
    seed = torch.zeros(2, 2)
    low, high = torch.full((2,), -1.0), torch.full((2,), 1.0)
    q = torch.zeros(2, 4, 2)
    pos = torch.tensor([[0.001, 0.05, 0.002, 0.003], [0.05, 0.05, 0.05, 0.05]])  # row 1 has no valid start
    rot = torch.full((2, 4), 0.1)
    drawn = set()
    gen = torch.Generator().manual_seed(0)
    for _ in range(40):
        pick = generator.select_ik_solution(
            q, pos, rot, seed, low, high, pos_tol=0.005, rot_tol=0.31, margin=0.03, randomize=True, generator=gen
        )
        assert pick.valid.tolist() == [True, False]
        assert pick.index[1] == 0  # no valid start: still the smallest position error, not a random one
        drawn.add(int(pick.index[0]))
    assert drawn == {0, 2, 3}  # never the invalid start 1, and the nearest start does not always win


def test_vertical_jaw_orientation_points_the_insert_axis_down_with_the_smallest_change_to_the_recording():
    goal = load_grasp_goal()
    rest = torch.tensor(scene.BAG_REST_ROT)[None]
    snapped = torch.tensor(generator.vertical_jaw_quat_bag(goal.jaw_quat_bag))[None].float()
    recorded = torch.tensor(goal.jaw_quat_bag)[None].float()
    axis = torch.tensor(JAW_INSERT_AXIS)[None].float()

    def insert_in_world(jaw_quat_bag):
        return quat_apply(rest, quat_apply(jaw_quat_bag, axis))

    assert insert_in_world(recorded)[0, 2] > -0.99  # the recording leans, which is what the rule removes
    assert insert_in_world(snapped)[0].tolist() == pytest.approx([0.0, 0.0, -1.0], abs=1e-5)
    # A tilt about a horizontal axis only: the rotation between them equals the lean of the recorded insert axis.
    lean = math.acos(float(-insert_in_world(recorded)[0, 2]))
    turn = 2.0 * math.acos(min(1.0, abs(float((snapped * recorded).sum()))))
    assert turn == pytest.approx(lean, abs=1e-4)


##
# Task configuration.
##


def _resolve(task: str, *presets: str):
    overrides = [f"presets={','.join(presets)}"] if presets else []
    return resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=overrides)


def test_the_arm_stands_upright_on_the_tray_wall():
    cfg, _ = _resolve("Isaac-HandleGrab")
    # init_state.rot is written onto the articulation root, so it carries the 90 deg roll and the 180 deg yaw.
    spawned = torch.tensor(cfg.scene.robot.init_state.rot)
    cad_up = quat_apply(spawned, torch.tensor((0.0, 1.0, 0.0)))
    assert torch.allclose(cad_up, torch.tensor((0.0, 0.0, 1.0)), atol=1e-5)
    assert cfg.scene.robot.init_state.rot == (0.0, 0.70710678, 0.70710678, 0.0)
    assert cfg.scene.robot.init_state.rot == scene.ARM_BASE_ROT
    assert cfg.scene.robot.init_state.pos == scene.ARM_BASE_POS
    assert scene.ARM_CFG.init_state.rot == scene.ARM_BASE_ROT
    assert cfg.scene.sandbox.spawn.usd_path == scene.SANDBOX_SPAWN_USD
    assert cfg.scene.bag.spawn.usd_path == scene.BAG_SPAWN_USD


def test_sim2real_preset_starts_every_episode_at_home_and_randomizes_the_physics():
    nominal, _ = _resolve("Isaac-HandleGrab")
    sim2real, _ = _resolve("Isaac-HandleGrab", "sim2real")
    assert sim2real.events.reset_from_dataset.params["phase_weights"] == {HOME: 1.0}
    assert {"jaw_material", "gripper_gains", "joint_parameters", "encoder_bias"} <= set(vars(sim2real.events))
    assert not {"jaw_material", "encoder_bias"} & set(vars(nominal.events))


@pytest.mark.parametrize(
    ("task", "entry_point", "groups"),
    [
        ("Isaac-HandleGrab", "rsl_rl_cfg_entry_point", {"actor": "policy", "critic": "critic"}),
        ("Isaac-HandleGrab-Camera", "rsl_rl_cfg_entry_point", {"actor": "overhead_depth", "critic": "critic"}),
        ("Isaac-HandleGrab-Camera", "rsl_rl_distillation_cfg_entry_point", {"student": "overhead_depth"}),
    ],
)
def test_agent_observation_groups_exist_in_the_environment(task, entry_point, groups):
    env_cfg, agent_cfg = resolve_task_config(task, entry_point, overrides=[])
    available = set(vars(env_cfg.observations))
    for role, example in groups.items():
        listed = agent_cfg.obs_groups[role]
        assert example in listed
        assert set(listed) <= available, f"{role} group {listed} is not a subset of {sorted(available)}"
    if "teacher" in agent_cfg.obs_groups:
        assert set(agent_cfg.obs_groups["teacher"]) <= available


##
# Evaluation.
##


class _ScriptedEnv:
    """Replays terminal statistics.

    Env 0 succeeds at step 2, then fails at step 4; env 1 drops the bag at step 5.
    """

    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.device, self.max_episode_length = 2, "cpu", 10
        self.termination_manager = SimpleNamespace(active_terms=["success"], get_term=self._term)
        self.step_count = 0

    def reset(self):
        return {}, {}

    def _term(self, name):
        dropped = self.step_count == 5
        return torch.tensor([False, dropped]) if name == "bag_dropped" else torch.zeros(2, dtype=torch.bool)

    def step(self, _actions):
        self.step_count += 1
        succeeded = self.step_count == 2
        done = torch.tensor([self.step_count in (2, 4), self.step_count == 5])
        self._hg_terminal_progress = torch.tensor([[True, succeeded], [True, False]])
        self._hg_terminal_max_lower_force = torch.tensor([0.5, 2.0])
        self._hg_terminal_time_to_success_s = torch.tensor([1.5 if succeeded else -0.03, -0.03])
        return {}, torch.zeros(2), done, torch.zeros(2, dtype=torch.bool), {}


def test_evaluation_counts_each_environments_first_episodes_only():
    summary = evaluation.evaluate(_ScriptedEnv(), lambda obs: torch.zeros(2, 1), episodes_per_env=1)
    # Env 0's failed second episode (step 4) finishes before env 1's first, and must not be counted.
    assert summary.episodes == 2
    assert summary.success_rate == 0.5 and summary.drop_rate == 0.5 and summary.grasp_rate == 1.0
    assert summary.mean_time_to_success_s == pytest.approx(1.5)
    assert summary.max_peak_lower_body_force_n == 2.0
