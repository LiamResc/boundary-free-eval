"""The boundary gap: how much it is worth to the adaptive controller to be told when friction or mass changes.

    uv run python playground/15_boundary_gap.py

Runs the experiment in src/boundary_free/experiment.py at each noise level: tunes FORGETTING for told and not
told on training streams, then compares them stream by stream on fresh test streams. Prints the results, then
draws two things: regret over time on one stream for both conditions, and the gap on every test stream. Saves
the picture to outputs/boundary_gap.png and opens it (add --no-show to skip that). Several minutes on all cores.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.controllers import adaptive_controller, static_controller, still_speed_for
from boundary_free.experiment import TEST_STREAMS, TRAINING_STREAMS, boundary_gap
from boundary_free.regret import measure_regret
from boundary_free.stream import example_schedule

# ---- Things to change and re-run ---------------------------------------------
NOISE_LEVELS = (0.0, 0.005, 0.01)  # m/s of speed-reading noise: none (the control), the main level, the hard case
EXAMPLE_NOISE = 0.005              # the noise for the one-stream picture of regret over time
DRIFT_EVERY = None                 # None: the drift is announced at its start and end; 2.0: every 2 s as well
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "boundary_gap.png"
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOUR = {"not told": "#1baf7a", "told": "#0c5e41"}  # adaptive's hue: lighter not told, darker told


def main():
    parser = argparse.ArgumentParser(description="Measure the boundary gap.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()
    n_runs = len(NOISE_LEVELS) * 2 * (6 * len(TRAINING_STREAMS) + len(TEST_STREAMS))
    print(f"Running {n_runs} streams over {os.cpu_count()} cores (several minutes) ...")

    with ProcessPoolExecutor() as pool:
        # 1. The experiment: tune each condition on the training streams, compare on the test streams.
        results = boundary_gap(NOISE_LEVELS, pool, drift_every=DRIFT_EVERY)
        # 2. One example stream, with checkpoints every 0.5 s, for the picture of regret over time.
        tuned = results[EXAMPLE_NOISE].forgetting
        example = dict(zip(("not told", "told", "static"),
                           pool.map(example_regret, ("not told", "told", "static"),
                                    (tuned["not told"], tuned["told"], None))))

    print_results(results)
    draw(results, example, tuned)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def example_regret(condition, forgetting):
    """The full regret record on the example stream (seed 0) at EXAMPLE_NOISE, for one condition."""
    still = still_speed_for(EXAMPLE_NOISE)
    if condition == "static":
        def make(world):
            return static_controller(still_speed=still)
    else:
        def make(world):
            return adaptive_controller(world, forgetting=forgetting, still_speed=still)
    return measure_regret(example_schedule(seed=0, speed_noise=EXAMPLE_NOISE), make, told=(condition == "told"),
                          drift_every=DRIFT_EVERY)


def print_results(results):
    print(f"\nAdaptive's regret rate on {len(TEST_STREAMS)} test streams (cm), each condition with its own FORGETTING, "
          f"tuned on {len(TRAINING_STREAMS)} training streams")
    print(f"{'noise':>8s} {'FORGETTING':>16s} {'not told':>9s} {'told':>7s} {'gap: mean [95% interval]':>27s} "
          f"{'median':>7s} {'told better':>12s} {'worst stream':>16s}")
    for noise, r in results.items():
        low, high = r.gap_interval()
        print(f"{f'{1000 * noise:g} mm/s':>8s} {r.forgetting['not told']:>7g} / {r.forgetting['told']:<7g}"
              f"{100 * r.not_told.mean():9.3f} {100 * r.told.mean():7.3f} "
              f"{f'{100 * r.gap.mean():+.3f} [{100 * low:+.3f}, {100 * high:+.3f}]':>27s} {100 * np.median(r.gap):+7.3f}"
              f" {f'{round(r.told_better * len(r.gap))} of {len(r.gap)}':>12s}"
              f" {f'{100 * r.not_told.max():.2f} / {100 * r.told.max():.2f}':>16s}")
    print("(gap = not told minus told, so positive means being told helped; worst stream: not told / told)")


def draw(results, example, tuned):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.2, 1]},
                                      facecolor=SURFACE)

    # Top: regret over time on one stream, for adaptive in both conditions, with the hidden changes marked and
    # a mark at each moment the told condition hears "something just changed".
    schedule = example_schedule(seed=0, speed_noise=EXAMPLE_NOISE)
    top.axhline(0.0, color=AXIS, linewidth=1.0)
    for condition in ("not told", "told"):
        record = example[condition]
        top.plot(record.times, 100 * np.array(record.regret), color=COLOUR[condition], linewidth=1.6,
                 label=f"adaptive, {condition} (FORGETTING {tuned[condition]:g}): {100 * record.rate:.2f} cm on average")
    for change in schedule.friction_changes + schedule.mass_changes:
        if change.over > 0:
            top.axvspan(change.at, change.at + change.over, color=MUTED, alpha=0.10, linewidth=0)
        else:
            top.axvline(change.at, color=AXIS, linewidth=1.0)
    moments = schedule.change_moments(DRIFT_EVERY)
    top.plot(moments, np.full(len(moments), 0.96), "v", color=COLOUR["told"], markersize=6,
             transform=top.get_xaxis_transform(), label="told: \"something just changed\"")
    top.set_xlim(0, schedule.length)
    top.set_xlabel("time (s)")
    top.set_ylabel("regret (cm)")
    top.set_title(f"One stream at {1000 * EXAMPLE_NOISE:g} mm/s of speed noise (static, never adapting, averages "
                  f"{100 * example['static'].rate:.2f} cm, off this scale)", loc="left", color=INK_SOFT, fontsize=9.5)
    top.legend(loc="center right", frameon=False, labelcolor=INK_SOFT, fontsize=8.5)

    # Bottom: the gap on every test stream at each noise level. Each dot is one stream: not told's regret minus
    # told's, so a dot above zero is a stream where being told helped. The bar is the median.
    rng = np.random.default_rng(0)
    bottom.axhline(0.0, color=AXIS, linewidth=1.0)
    for i, (noise, r) in enumerate(results.items()):
        gap = 100 * r.gap
        bottom.plot(i + rng.uniform(-0.18, 0.18, len(gap)), gap, "o", color=INK_SOFT, alpha=0.55, markersize=4,
                    markeredgewidth=0)
        bottom.plot([i - 0.25, i + 0.25], [np.median(gap)] * 2, color=COLOUR["told"], linewidth=2.5)
        bottom.annotate(f"told better on {round(r.told_better * len(gap))} of {len(gap)}\nmedian {np.median(gap):+.3f} cm",
                        xy=(i + 0.28, np.median(gap)), va="center", color=INK_SOFT, fontsize=8.5)
    bottom.set_xticks(range(len(results)), [f"{1000 * n:g} mm/s" for n in results])
    bottom.set_xlim(-0.5, len(results) - 0.2)
    bottom.set_xlabel("speed noise")
    bottom.set_ylabel("gap: not told minus told (cm)")
    bottom.set_title(f"Every test stream ({len(TEST_STREAMS)} per noise level): above zero, being told helped",
                     loc="left", color=INK_SOFT, fontsize=9.5)

    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_SOFT, labelsize=9)
        ax.yaxis.label.set_color(INK_SOFT)
        ax.xaxis.label.set_color(INK_SOFT)

    fig.suptitle("The boundary gap: what being told \"something just changed\" is worth to the adaptive controller",
                 x=0.07, ha="left", color=INK, fontsize=12.5)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.07, hspace=0.42)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
