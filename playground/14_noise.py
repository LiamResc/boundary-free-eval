"""Noise and memory: how adaptive's regret depends on how long it remembers, at several levels of speed noise.

    uv run python playground/14_noise.py

Measures adaptive's regret on the example stream for every pair of noise level and FORGETTING value below, each
with a few different noise draws, prints the table, then draws regret against memory, one line per noise level,
saves the picture to outputs/noise.png and opens it in a window (add --no-show to skip that). The runs are spread
over the computer's cores; about half a minute on a recent laptop.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.controllers import adaptive_controller, static_controller
from boundary_free.regret import measure_regret
from boundary_free.stream import Schedule, example_schedule

# ---- Things to change and re-run ---------------------------------------------
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)           # m/s of noise on each speed reading (0.005 is 5 mm/s)
FORGETTINGS = (0.9995, 0.999, 0.998, 0.995, 0.99)  # per 2 ms step: memory of about 4, 2, 1, 0.4 and 0.2 s of sliding
NOISE_DRAWS = (0, 1, 2, 3, 4)                       # noise seeds: the same noise level, drawn five different ways
WINDOW = 25                                         # sliding steps per learning point (50 ms)
CHECK_EVERY = 2.0                                   # seconds between regret checkpoints (sparser than 12, for speed)
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "noise.png"
STEP = 0.002  # seconds per simulation step, for turning FORGETTING into a memory in seconds

# Colours: one hue (adaptive's) from light to dark as the noise grows; everything else in neutral greys.
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ("#9fdcc3", "#4fbf95", "#1b9b6c", "#0c5e41")


def main():
    parser = argparse.ArgumentParser(description="Adaptive's regret against memory, at several noise levels.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    # 1. Every run to make: each noise level with each FORGETTING, under each noise draw. With no noise, every
    #    draw is the same, so one is enough.
    jobs = [(noise, forgetting, draw) for noise in NOISE_LEVELS for forgetting in FORGETTINGS
            for draw in (NOISE_DRAWS if noise > 0 else NOISE_DRAWS[:1])]
    print(f"Measuring regret for {len(jobs)} runs, spread over {os.cpu_count()} cores ...")

    # 2. Measure them all, several at a time; plus static, once per noise level, for comparison.
    with ProcessPoolExecutor() as pool:
        rates = dict(zip(jobs, pool.map(adaptive_regret, *zip(*jobs))))
        static = dict(zip(NOISE_LEVELS, pool.map(static_regret, NOISE_LEVELS)))

    # 3. Print the table, then draw it.
    print_table(rates, static)
    draw(rates, static)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def schedule(noise, draw):
    """The example stream (friction 0.3 -> 0.6 -> 0.3, mass 1 -> 1.5) with this much speed noise, drawn this way."""
    example = example_schedule()
    return Schedule(targets=example.targets, target_every=example.target_every, friction_start=example.friction_start,
                    mass_start=example.mass_start, friction_changes=example.friction_changes,
                    mass_changes=example.mass_changes, speed_noise=noise, noise_seed=draw)


def adaptive_regret(noise, forgetting, draw):
    record = measure_regret(schedule(noise, draw), lambda world: adaptive_controller(world, forgetting=forgetting,
                                                                                     window=WINDOW),
                            check_every=CHECK_EVERY)
    return record.rate


def static_regret(noise):
    return measure_regret(schedule(noise, 0), lambda world: static_controller(), check_every=CHECK_EVERY).rate


def memory(forgetting):
    """About how many seconds of sliding a FORGETTING value remembers."""
    return STEP / (1 - forgetting)


def summary(rates, noise, forgetting):
    """Adaptive's regret (cm) for one noise level and FORGETTING: the average over the draws, the lowest, the highest."""
    values = 100 * np.array([rate for (n, f, _), rate in rates.items() if n == noise and f == forgetting])
    return values.mean(), values.min(), values.max()


def print_table(rates, static):
    print("\nAdaptive's regret rate (cm): the average over the noise draws, with the lowest and highest in brackets")
    print(f"{'speed noise':>14s}" + "".join(f"{f'{memory(f):g} s memory':>22s}" for f in FORGETTINGS) + f"{'static':>10s}")
    for noise in NOISE_LEVELS:
        cells = []
        for forgetting in FORGETTINGS:
            mean, low, high = summary(rates, noise, forgetting)
            cells.append(f"{mean:8.2f} ({low:.2f}-{high:.2f})".rjust(22))
        print(f"{f'{1000 * noise:g} mm/s':>14s}" + "".join(cells) + f"{100 * static[noise]:10.2f}")
    print(f"{'FORGETTING':>14s}" + "".join(f"{f:>22g}" for f in FORGETTINGS))


def draw(rates, static):
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    memories = [memory(f) for f in FORGETTINGS]
    stats = {noise: np.array([summary(rates, noise, f) for f in FORGETTINGS]) for noise in NOISE_LEVELS}
    # Anything worse than static, which never adapts at all, is drawn at the top edge with its value written on,
    # so one disaster does not squash every other line flat.
    static_level = 100 * np.mean(list(static.values()))
    top = 1.25 * max(mean for s in stats.values() for mean in s[:, 0] if mean < static_level)
    for noise, colour in zip(NOISE_LEVELS, RAMP):
        means = stats[noise][:, 0]
        ax.fill_between(memories, stats[noise][:, 1], stats[noise][:, 2], color=colour, alpha=0.18, linewidth=0)
        ax.plot(memories, np.minimum(means, top), color=colour, linewidth=1.8, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{1000 * noise:g} mm/s")
        best = int(np.argmin(means))
        ax.plot([memories[best]], [means[best]], marker="o", markersize=10, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.2)
        for m, mean in zip(memories, means):
            if mean > top:
                ax.annotate(f"{1000 * noise:g} mm/s: {mean:.1f} cm, worse than static", xy=(m, top), xytext=(-8, -4),
                            textcoords="offset points", ha="right", va="top", color=INK_SOFT, fontsize=8.5)
    ax.set_ylim(0, top)
    ax.set_xscale("log")
    ax.set_xticks(memories, [f"{m:g} s" for m in memories])
    ax.minorticks_off()
    ax.set_xlabel("memory: about how many seconds of sliding it remembers (longest on the left)")
    ax.set_ylabel("adaptive's regret rate (cm)")
    ax.invert_xaxis()  # longest memory on the left, shortest on the right, as FORGETTING falls
    ax.legend(title="speed noise", loc="upper left", frameon=False, labelcolor=INK_SOFT, fontsize=9, title_fontsize=9)

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
    ax.get_legend().get_title().set_color(INK_SOFT)

    fig.suptitle("With noise, the best memory is neither the longest nor the shortest", x=0.08, ha="left",
                 color=INK, fontsize=12.5)
    fig.text(0.08, 0.925, f"Lines: the average over {len(NOISE_DRAWS)} noise draws; bands: lowest to highest; circled: "
             f"the best memory at each noise level.\nStatic's regret is about {static_level:.1f} cm at every noise "
             "level, far above this chart.", ha="left", va="top", color=INK_SOFT, fontsize=9, linespacing=1.4)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.11)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
