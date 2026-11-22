"""Deviation regret on the arm: static and the three adaptive versions, each measured against the oracle along the
same arm stream.

    uv run python playground/23_arm_regret.py

At every checkpoint the arm world is branched and the oracle takes over (inheriting whatever the controller is in the
middle of, then deciding everything with the true friction); its next 4 s are scored against what the controller
actually did. Prints each controller's regret rate overall and per stretch of the schedule, then draws the hidden
friction above each controller's regret over time, saves the picture to outputs/arm_regret.png and opens it in a
window (add --no-show to skip that). The four runs go in parallel: under a minute.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.arm_stream import arm_schedule, measure_arm_regret
from boundary_free.shove_controllers import AdaptiveShover, static_shover

# ---- Things to change and re-run ---------------------------------------------
SPEED_NOISE = 0.01       # m/s of noise on each speed reading
POSITION_NOISE = 0.002   # m of noise on each position reading
FORGETTING = 0.95        # the adaptive versions' forgetting per reading
CHECK_EVERY = 1.0        # seconds between checkpoints
SEED = 0                 # which targets and which noise
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "arm_regret.png"
CONTROLLERS = ("static", "forgetting only", "self-detected", "told")
STRETCHES = ((0, 39), (39, 63), (63, 75), (75, 81), (81, 111), (111, 152))
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LINE = "#2a78d6"


def measure(name):
    schedule = arm_schedule(SEED, speed_noise=SPEED_NOISE, position_noise=POSITION_NOISE)
    if name == "static":
        def make(world, sensor):
            return static_shover(world, sensor=sensor)
    else:
        def make(world, sensor):
            return AdaptiveShover(world, trigger=name, forgetting=FORGETTING, sensor=sensor)
    record = measure_arm_regret(schedule, make, told=(name == "told"), check_every=CHECK_EVERY)
    return name, np.array(record.times), 100 * np.array(record.regret)  # centimetres


def main():
    parser = argparse.ArgumentParser(description="Regret against the oracle on the arm stream.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    with ProcessPoolExecutor() as pool:
        results = {name: (times, regret) for name, times, regret in pool.map(measure, CONTROLLERS)}
    print(f"speed noise {1000 * SPEED_NOISE:g} mm/s, position noise {1000 * POSITION_NOISE:g} mm, forgetting "
          f"{FORGETTING}, a checkpoint every {CHECK_EVERY:g} s\n")
    print("regret rate (cm)        overall   " + "  ".join(f"{a}-{b} s".rjust(9) for a, b in STRETCHES))
    for name in CONTROLLERS:
        times, regret = results[name]
        cells = "  ".join(f"{regret[(times >= a) & (times < b)].mean():9.3f}" for a, b in STRETCHES)
        print(f"  {name:<20s} {regret.mean():7.3f}   {cells}")
    draw(results)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def draw(results):
    schedule = arm_schedule(SEED)
    fig, axes = plt.subplots(len(CONTROLLERS) + 1, 1, figsize=(10, 8.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": (0.8,) + (1,) * len(CONTROLLERS)})
    t = np.linspace(0, schedule.length, 1000)
    axes[0].plot(t, [schedule.friction(x) for x in t], color=MUTED, linewidth=1.6)
    axes[0].set_ylabel("hidden\nfriction")
    top = max(regret.max() for _, regret in results.values())
    bottom = min(0.0, min(regret.min() for _, regret in results.values()))
    for ax, name in zip(axes[1:], CONTROLLERS):
        times, regret = results[name]
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.plot(times, regret, color=LINE, linewidth=1.4)
        ax.set_ylim(bottom - 0.05 * (top - bottom), top * 1.1)
        ax.set_ylabel("regret\n(cm)")
        ax.annotate(f"{name}: rate {regret.mean():.3f} cm", xy=(0.01, 0.92), xycoords="axes fraction", ha="left",
                    va="top", color=INK, fontsize=9)
    for ax in axes:
        for moment in schedule.change_moments():
            ax.axvline(moment, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_SOFT, labelsize=9)
        ax.yaxis.label.set_color(INK_SOFT)
    axes[-1].set_xlabel("time (s)")
    axes[-1].xaxis.label.set_color(INK_SOFT)
    fig.suptitle("Regret against the oracle on the arm stream", x=0.08, ha="left", color=INK, fontsize=12.5)
    fig.text(0.08, 0.945, f"Same targets and noise for all; dashed lines mark the hidden changes. Speed noise "
             f"{1000 * SPEED_NOISE:g} mm/s, forgetting {FORGETTING}.", ha="left", va="top", color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.07, hspace=0.15)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
