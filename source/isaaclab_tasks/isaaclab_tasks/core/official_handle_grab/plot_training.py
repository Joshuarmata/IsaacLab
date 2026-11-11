# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot a handle-grab training run from its TensorBoard log: outcomes, grasp rate, every reward term, and PPO health.

Iterations are on every x axis. ``Episode_Reward/<term>`` is what Isaac Lab logs: the episode's sum of that term
(``value x weight x step_dt``) divided by the episode's maximum length in seconds, so a reward per second; pass
``--per_episode`` to multiply it back to a per-episode return. ``Episode_Termination/<term>`` is the fraction of the
episodes that ended in an iteration that ended for that reason, so ``Episode_Termination/success`` is the per-episode
success rate over every start; ``Curriculum/success_by_start/<phase>`` splits it by the reset phase the episode started
from, and ``home`` is what play and evaluation measure. ``Metrics/grasp_rate`` is the fraction of environments whose
grasp has latched, sampled every step::

    uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training
    uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training \
        --run 2026-11-10_15-16-00 --compare 2026-11-10_13-58-15

Writes ``training_curves.png`` into the (first) run directory unless ``--output`` is given. ``--live`` instead opens a
window that redraws every ``--refresh`` seconds while ``train.py`` runs, following the newest run unless ``--run``
is given::

    uv run python -m isaaclab_tasks.core.official_handle_grab.plot_training --live

Runs are looked up under ``logs/rsl_rl/handle_grab_state`` unless ``--experiment`` names another, such as
``handle_grab_camera_distillation`` for the camera student, whose ``Loss/behavior`` is plotted with the noise std.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LOG_ROOT = Path("logs/rsl_rl")
"""Where ``train.py`` writes runs, one directory per experiment, relative to the repository root."""
EXPERIMENTS = ("handle_grab_state", "handle_grab_camera_distillation", "handle_grab_camera")
"""This task's experiments: the state teacher, the camera student, and the camera policy trained from scratch."""
OUTCOMES = ("success", "time_out", "bag_dropped", "bag_out_of_bounds", "unstable_robot")
"""Termination terms, plotted as the fraction of finished episodes that ended for each."""
SUCCESS_TERM = "success"
"""Reward term plotted on its own: its bonus would flatten every other reward curve."""
MILESTONES = (0.25, 0.5, 0.75, 0.9)
"""Per-episode success rates whose first iteration the summary reports."""


def _resolve_run(name: str | None, experiment: str) -> Path:
    """A run directory: a path, a run name under the experiment's log directory, or its newest run when ``None``."""
    root = LOG_ROOT / experiment
    if name is None:
        runs = sorted((p for p in root.glob("*") if p.is_dir()), key=lambda p: p.name)
        if not runs:
            raise SystemExit(f"No runs under {root}. Run from the repository root, or pass --run.")
        return runs[-1]
    path = Path(name)
    if not path.exists():
        path = root / name
    if not path.is_dir():
        raise SystemExit(f"Run directory '{name}' not found (also tried {root / name}).")
    return path


def load_scalars(run: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every scalar of a run as ``{tag: (iterations, values)}``, merged over its event files in step order."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(run.glob("events.out.tfevents.*"))
    if not files:
        raise SystemExit(f"No TensorBoard event files in {run}.")
    merged: dict[str, dict[int, float]] = {}
    for file in files:
        acc = EventAccumulator(str(file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            merged.setdefault(tag, {}).update({e.step: e.value for e in acc.Scalars(tag)})
    return {
        tag: (np.array(sorted(points)), np.array([points[s] for s in sorted(points)])) for tag, points in merged.items()
    }


def _episode_seconds(run: Path) -> float:
    """``episode_length_s`` from the run's saved environment config, the divisor of ``Episode_Reward``."""
    env_yaml = run / "params" / "env.yaml"
    if env_yaml.exists():
        for line in env_yaml.read_text().splitlines():
            if line.startswith("episode_length_s:"):
                return float(line.split(":", 1)[1])
    return 20.0


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over ``window`` iterations (shorter at the start)."""
    if window <= 1 or values.size == 0:
        return values
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    end = np.arange(1, values.size + 1)
    start = np.maximum(end - window, 0)
    return (cumsum[end] - cumsum[start]) / (end - start)


def _first_reaching(steps: np.ndarray, values: np.ndarray, level: float) -> int | None:
    hit = np.nonzero(values >= level)[0]
    return int(steps[hit[0]]) if hit.size else None


def summarize(name: str, scalars: dict[str, tuple[np.ndarray, np.ndarray]], window: int) -> None:
    """Print where the run started and ended, and how quickly per-episode success climbed."""
    print(f"\n{name}")
    rows = [
        ("Episode_Termination/success", "per-episode success"),
        ("Episode_Termination/time_out", "per-episode time-out"),
        ("Curriculum/success_by_start/home", "success from home"),
        ("Metrics/grasp_rate", "grasp rate"),
        ("Train/mean_reward", "mean reward"),
        ("Train/mean_episode_length", "episode length [steps]"),
        ("Policy/mean_std", "policy noise std"),
        ("Loss/behavior", "student-teacher loss"),
    ]
    for tag, label in rows:
        if tag in scalars:
            steps, values = scalars[tag]
            smoothed = smooth(values, window)
            first, last = f"iteration {steps[0]}: {smoothed[0]:8.3f}", f"iteration {steps[-1]}: {smoothed[-1]:8.3f}"
            print(f"  {label:24s} {first}   {last}")
    if "Episode_Termination/success" in scalars:
        steps, values = scalars["Episode_Termination/success"]
        smoothed = smooth(values, window)
        reached = ", ".join(
            f"{level:.0%} at {it if (it := _first_reaching(steps, smoothed, level)) is not None else 'never'}"
            for level in MILESTONES
        )
        print(f"  per-episode success first reaches (smoothed): {reached}")


FIGURE_SIZE = (15, 17)


def draw_figure(fig, data: list[tuple[Path, dict]], window: int, per_episode: bool) -> None:
    """Clear ``fig`` and draw every panel for ``data``, a list of ``(run, scalars)``."""
    import matplotlib.pyplot as plt

    fig.clear()
    styles = ("-", "--", ":", "-.")
    axes = fig.subplots(4, 2, sharex=True)
    (ax_out, ax_start), (ax_grasp, ax_cur), (ax_pos, ax_success), (ax_neg, ax_ppo) = axes
    reward_unit = "reward per episode" if per_episode else "reward per second (Isaac Lab Episode_Reward)"
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    twin = None
    twin_return = None

    for k, (run, scalars) in enumerate(data):
        style = styles[k % len(styles)]
        label_run = "" if len(data) == 1 else f" [{run.name}]"
        scale = _episode_seconds(run) if per_episode else 1.0

        def draw(ax, tag, label, color, factor=1.0, _scalars=scalars, _style=style, _label_run=label_run):
            if tag not in _scalars:
                return
            steps, values = _scalars[tag]
            ax.plot(steps, smooth(values * factor, window), _style, color=color, label=label + _label_run)

        for i, term in enumerate(OUTCOMES):
            draw(ax_out, f"Episode_Termination/{term}", term, colors[i % len(colors)])
        draw(ax_grasp, "Metrics/grasp_rate", "grasp rate (latched)", colors[0])
        draw(ax_grasp, "Metrics/success_rate", "success this step", colors[1])

        rewards = sorted(t.split("/", 1)[1] for t in scalars if t.startswith("Episode_Reward/"))
        # A term that never paid (mean 0) is a reward that was not reached, not a penalty.
        negative = [t for t in rewards if t != SUCCESS_TERM and scalars[f"Episode_Reward/{t}"][1].mean() < 0]
        positive = [t for t in rewards if t != SUCCESS_TERM and t not in negative]
        for i, term in enumerate(positive):
            draw(ax_pos, f"Episode_Reward/{term}", term, colors[i % len(colors)], scale)
        for i, term in enumerate(negative):
            draw(ax_neg, f"Episode_Reward/{term}", term, colors[i % len(colors)], scale)
        draw(ax_success, f"Episode_Reward/{SUCCESS_TERM}", "success bonus", colors[0], scale)
        if "Train/mean_reward" in scalars:
            twin_return = twin_return or ax_success.twinx()
            steps, values = scalars["Train/mean_reward"]
            twin_return.plot(steps, smooth(values, window), style, color=colors[1], label="mean return" + label_run)
            twin_return.set_ylabel("mean return per episode (Train/mean_reward)")
        draw(ax_ppo, "Policy/mean_std", "policy noise std", colors[0])
        draw(ax_ppo, "Loss/behavior", "student-teacher action loss", colors[2])
        for i, tag in enumerate(t for t in sorted(scalars) if t.startswith("Curriculum/success_by_start/")):
            draw(ax_start, tag, f"from {tag.rsplit('/', 1)[1]}", colors[i % len(colors)])
        draw(ax_cur, "Curriculum/reset_fade", "grasp_closing start weight (others 1)", colors[0])
        draw(ax_cur, "Curriculum/lower_body_ramp", "lower_body_contact weight", colors[1])
        if "Train/mean_episode_length" in scalars:
            twin = twin or ax_ppo.twinx()
            steps, values = scalars["Train/mean_episode_length"]
            twin.plot(steps, smooth(values, window), style, color=colors[3], label="episode length" + label_run)
            twin.set_ylabel("episode length [policy steps]")

    ax_out.set_title("How episodes end (fraction of finished episodes)")
    ax_out.set_ylabel("fraction")
    ax_out.set_ylim(-0.02, 1.02)
    ax_grasp.set_title("Grasp (holding latched), fraction of environments")
    ax_grasp.set_ylabel("fraction")
    ax_grasp.set_ylim(-0.02, 1.02)
    ax_pos.set_title("Positive reward terms")
    ax_pos.set_ylabel(reward_unit)
    ax_success.set_title("Success bonus (left) and total return (right)")
    ax_success.set_ylabel(reward_unit)
    ax_neg.set_title("Penalty terms")
    ax_neg.set_ylabel(reward_unit)
    ax_ppo.set_title("Exploration noise, distillation loss, and episode length")
    ax_ppo.set_ylabel("std / loss")
    ax_start.set_title("Success by start type (running average of finished episodes)")
    ax_start.set_ylabel("fraction")
    ax_start.set_ylim(-0.02, 1.02)
    ax_cur.set_title("Curriculum: reset mix and penalty ramp")
    ax_cur.set_ylabel("weight")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        right = {id(ax_ppo): twin, id(ax_success): twin_return}.get(id(ax))
        if right is not None:
            more = right.get_legend_handles_labels()
            handles, labels = handles + more[0], labels + more[1]
        if handles:
            ax.legend(handles, labels, fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("PPO iteration")
    names = ", ".join(run.name for run, _ in data)
    last = max((int(steps[-1]) for _, scalars in data for steps, _ in scalars.values() if steps.size), default=0)
    fig.suptitle(
        f"Isaac-HandleGrab training: {names}, up to iteration {last} (moving average over {window} iterations)"
    )
    fig.tight_layout()


def plot(runs: list[Path], window: int, per_episode: bool, output: Path, show: bool) -> None:
    """Draw the runs once, save the image, and print the summary."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [(run, load_scalars(run)) for run in runs]
    fig = plt.figure(figsize=FIGURE_SIZE)
    draw_figure(fig, data, window, per_episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    print(f"\nwrote {output}")
    for run, scalars in data:
        summarize(run.name, scalars, window)
    if show:
        plt.show()


def live(run: str | None, compare: list[str], experiment: str, window: int, per_episode: bool, refresh: float) -> None:
    """Keep a window open and redraw it every ``refresh`` seconds from the growing TensorBoard log.

    Without ``run`` it follows the experiment's newest run, so it can be started before ``train.py``: it
    switches to the new run as soon as its directory appears. Close the window to stop. A redraw resets zoom.
    """
    import time

    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(figsize=FIGURE_SIZE)
    shown = None
    while plt.fignum_exists(fig.number):
        try:
            runs = [_resolve_run(run, experiment), *(_resolve_run(name, experiment) for name in compare)]
            data = [(r, load_scalars(r)) for r in runs]
        except SystemExit as exc:  # a run directory without an event file yet
            print(f"[live] waiting: {exc}", flush=True)
            plt.pause(refresh)
            continue
        if runs[0] != shown:
            print(f"[live] following {runs[0]}", flush=True)
            shown = runs[0]
        draw_figure(fig, data, window, per_episode)
        fig.canvas.draw_idle()
        print(f"[live] {time.strftime('%H:%M:%S')} redrawn; next in {refresh:.0f} s", flush=True)
        plt.pause(refresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a handle-grab training run from its TensorBoard log.")
    parser.add_argument(
        "--experiment", default=EXPERIMENTS[0], choices=EXPERIMENTS, help="Which log directory under logs/rsl_rl."
    )
    parser.add_argument("--run", default=None, help="Run directory, or run name under the experiment.")
    parser.add_argument("--compare", nargs="*", default=[], help="More runs to overlay (dashed, dotted, ...).")
    parser.add_argument("--smooth", type=int, default=10, help="Moving-average window [iterations]; 1 for raw.")
    parser.add_argument("--per_episode", action="store_true", help="Rewards per episode instead of per second.")
    parser.add_argument("--output", type=Path, default=None, help="Image (default <run>/training_curves.png).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    parser.add_argument("--live", action="store_true", help="Live window that redraws while training runs.")
    parser.add_argument("--refresh", type=float, default=30.0, help="Seconds between live redraws.")
    args = parser.parse_args(argv)
    if args.live:
        live(args.run, args.compare, args.experiment, args.smooth, args.per_episode, args.refresh)
        return 0
    runs = [_resolve_run(args.run, args.experiment), *(_resolve_run(name, args.experiment) for name in args.compare)]
    plot(runs, args.smooth, args.per_episode, args.output or runs[0] / "training_curves.png", args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
