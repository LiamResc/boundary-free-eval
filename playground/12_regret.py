"""Regret: at every checkpoint along the stream, how much better the oracle would have done from there.

    uv run python playground/12_regret.py

Measures the static controller's regret against the oracle along the example stream and prints it stretch by
stretch, then draws it over time, saves the picture to outputs/regret.png and opens it in a window
(add --no-show to skip that).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.controllers import static_controller
from boundary_free.regret import measure_regret
from boundary_free.stream import Change, Schedule, example_schedule

# ---- Things to change and re-run ---------------------------------------------
# The hidden changes, written as in 11_the_stream.py. Static believes friction 0.3 and mass 1.0 throughout.
FRICTION_CHANGES = (Change(at=39, to=0.6), Change(at=81, to=0.3))
MASS_CHANGES = (Change(at=63, to=1.5, over=12),)
# How regret is measured. These are the defaults in src/boundary_free/regret.py.
LOOK_AHEAD = 4.0   # seconds each checkpoint looks ahead
FADE_TIME = 1.0    # a moment this far ahead counts about a third as much as the present
CHECK_EVERY = 0.5  # seconds between checkpoints
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "regret.png"

# Colours: one per controller, everything else in neutral greys (as in 11_the_stream.py).
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOUR = {"static": "#2a78d6", "oracle": "#eb6834"}


def main():
    parser = argparse.ArgumentParser(description="Measure static's regret against the oracle along the stream.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    # 1. Measure: run static through the whole stream, and at every checkpoint let the oracle take over a branch.
    schedule = Schedule(targets=example_schedule().targets, target_every=6.0, friction_start=0.3, mass_start=1.0,
                        friction_changes=FRICTION_CHANGES, mass_changes=MASS_CHANGES)
    print(f"Measuring: at every checkpoint, the oracle runs {LOOK_AHEAD:g} s ahead on a branch ...")
    record = measure_regret(schedule, lambda world: static_controller(),
                            look_ahead=LOOK_AHEAD, fade_time=FADE_TIME, check_every=CHECK_EVERY)

    # 2. Print it: stretch by stretch, then for the whole stream.
    print_regret(schedule, record)

    # 3. Draw it.
    draw(schedule, record)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def stretches(schedule):
    """Split the stream at every moment a hidden change starts or ends."""
    moments = {0.0, schedule.length}
    for change in schedule.friction_changes + schedule.mass_changes:
        moments |= {change.at, change.at + change.over}
    moments = sorted(m for m in moments if 0.0 <= m <= schedule.length)
    return list(zip(moments[:-1], moments[1:]))


def describe(schedule, start, end):
    """Friction and mass over a stretch, such as 'friction 0.6, mass 1.0 -> 1.5'."""
    def knob(lookup):
        first, last = round(lookup(start + 1e-6), 2), round(lookup(end - 1e-6), 2)
        return f"{first:g}" if first == last else f"{first:g} -> {last:g}"
    return f"friction {knob(schedule.friction)}, mass {knob(schedule.mass)}"


def print_regret(schedule, record):
    times, regret = np.array(record.times), np.array(record.regret)
    print(f"\n{len(times)} checkpoints, every {CHECK_EVERY:g} s. Regret at a checkpoint: how much closer to the targets")
    print(f"the oracle would have kept the block over the next {LOOK_AHEAD:g} s (an average, nearer moments counting more).")
    print("\nAverage regret in each stretch of the stream, by checkpoint time (cm):")
    for start, end in stretches(schedule):
        during = (times >= start) & (times < end)
        cell = f"{100 * regret[during].mean():8.2f}" if during.any() else f"{'-':>8s}"  # '-': no checkpoints
        print(f"{start:5.0f} to {end:3.0f} s   {describe(schedule, start, end):38s}{cell}")
    print(f"\nTotal regret, added up over all {len(times)} checkpoints: {100 * record.total:.1f} cm")
    print(f"Regret rate, the average per checkpoint:          {100 * record.rate:.2f} cm")


def draw(schedule, record):
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 7.5),
                                      gridspec_kw={"height_ratios": [3, 2]}, facecolor=SURFACE)
    times = np.array(record.times)
    regret = 100 * np.array(record.regret)                  # in cm
    static_miss = -100 * np.array(record.controller_score)  # a score is minus an average miss
    oracle_miss = -100 * np.array(record.oracle_score)

    # Top: regret at each checkpoint.
    top.axhline(0.0, color=AXIS, linewidth=1.0)
    top.plot(times, regret, color=COLOUR["static"], linewidth=1.6)
    top.set_ylim(min(0.0, 1.3 * regret.min()), max(1.0, 1.35 * regret.max()))
    top.set_ylabel("regret (cm)")
    top.annotate("exactly 0 before the first change: static's beliefs\nare right, so it does just what the oracle would",
                 xy=(1.0, 0.12 * top.get_ylim()[1]), color=INK_SOFT, fontsize=8, va="bottom")

    # Bottom: where regret comes from. Each one's miss over the look-ahead; regret is the gap between them.
    bottom.fill_between(times, oracle_miss, static_miss, color=COLOUR["static"], alpha=0.12, linewidth=0,
                        label="the gap between them: regret")
    bottom.plot(times, static_miss, color=COLOUR["static"], linewidth=1.6, label="static, its own future")
    bottom.plot(times, oracle_miss, color=COLOUR["oracle"], linewidth=1.6, label="oracle, on a branch")
    bottom.set_ylabel(f"miss over the\nnext {LOOK_AHEAD:g} s (cm)")
    bottom.set_xlabel("checkpoint time (s)")
    bottom.set_xlim(0, schedule.length)
    bottom.legend(loc="lower left", frameon=False, ncols=3, labelcolor=INK_SOFT, fontsize=9,
                  bbox_to_anchor=(0.0, 1.0))
    if abs(static_miss[-1] - oracle_miss[-1]) > 0.08 * bottom.get_ylim()[1]:  # name the lines where they end apart
        for name, miss in (("static", static_miss), ("oracle", oracle_miss)):
            bottom.annotate(name, xy=(times[-1], miss[-1]), xytext=(6, 0), textcoords="offset points",
                            color=INK_SOFT, fontsize=8.5, va="center")

    # Mark each hidden change on both panels: a line for a jump, a light band for a drift.
    for knob, change in ([("friction", c) for c in schedule.friction_changes]
                         + [("mass", c) for c in schedule.mass_changes]):
        for ax in (top, bottom):
            if change.over > 0:
                ax.axvspan(change.at, change.at + change.over, color=MUTED, alpha=0.10, linewidth=0)
            else:
                ax.axvline(change.at, color=AXIS, linewidth=1.0)
        top.annotate(f"{knob} → {change.to:g}", xy=(change.at, top.get_ylim()[1]),
                     xytext=(4, -12), textcoords="offset points", color=INK_SOFT, fontsize=8.5)

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

    fig.suptitle("Static's regret against the oracle, along the stream", x=0.065, ha="left", color=INK, fontsize=13)
    fig.text(0.065, 0.925, f"At each checkpoint (every {CHECK_EVERY:g} s): how much closer the oracle would have kept "
             f"the block to its targets over the next {LOOK_AHEAD:g} s.\nFriction and mass change on a hidden schedule; "
             f"static never adapts. Average over the whole stream: {100 * record.rate:.2f} cm.",
             ha="left", va="top", color=INK_SOFT, fontsize=9.5, linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.08, hspace=0.22)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
