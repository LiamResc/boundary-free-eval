"""The stream: static and oracle on one long run, with hidden changes to friction and mass.

    uv run python playground/11_the_stream.py

Prints how each controller did in each stretch of the stream and which targets it reached, then draws
the whole run, saves the picture to outputs/stream.png and opens it in a window (add --no-show to skip that).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.controllers import OracleController, static_controller
from boundary_free.stream import Change, Schedule, Stream, example_schedule, run, targets_reached

# ---- Things to change and re-run ---------------------------------------------
# The hidden changes. Each is Change(at=seconds, to=new value, over=seconds the change takes; 0 is a jump).
# Static believes friction 0.3 and mass 1.0 throughout.
FRICTION_CHANGES = (Change(at=39, to=0.6), Change(at=81, to=0.3))
MASS_CHANGES = (Change(at=63, to=1.5, over=12),)
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "stream.png"

# Colours: one per controller, everything else in neutral greys.
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOUR = {"static": "#2a78d6", "oracle": "#eb6834"}


def main():
    parser = argparse.ArgumentParser(description="Run static and oracle on the example stream.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    # 1. Run both controllers on the same stream, start to finish: the example stream's targets,
    #    with the hidden changes set above.
    targets = example_schedule().targets
    schedule = Schedule(targets=targets, target_every=6.0, friction_start=0.3, mass_start=1.0,
                        friction_changes=FRICTION_CHANGES, mass_changes=MASS_CHANGES)
    records = {}
    for name in ("static", "oracle"):
        stream = Stream(schedule)
        controller = static_controller() if name == "static" else OracleController(stream.world)
        records[name] = run(stream, controller, schedule.length)

    # 2. Print how each did, stretch by stretch, and target by target.
    print_stretches(schedule, records)
    print_targets(schedule, records)

    # 3. Draw it.
    draw(schedule, records)
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


def print_stretches(schedule, records):
    print("Average miss in each stretch of the stream (cm):")
    print(f"{'':16s}{'':38s}{'static':>8s}{'oracle':>8s}")
    for start, end in stretches(schedule):
        cells = []
        for name in ("static", "oracle"):
            r = records[name]
            time, miss = np.array(r.time), np.abs(np.array(r.position) - np.array(r.target))
            during = (time > start) & (time <= end)
            cells.append(f"{100 * miss[during].mean():8.2f}")
        print(f"{start:5.0f} to {end:3.0f} s   {describe(schedule, start, end):38s}{''.join(cells)}")


def print_targets(schedule, records):
    print()
    print("Targets reached within 1 cm before the next one arrived (Y = reached, . = missed):")
    for name in ("static", "oracle"):
        reached = targets_reached(records[name], schedule)
        print(f"   {name:>6}: {''.join('Y' if r else '.' for r in reached)}   {sum(reached)} of {len(reached)}")


def draw(schedule, records):
    fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5),
                                              gridspec_kw={"height_ratios": [3, 1, 1]}, facecolor=SURFACE)
    time = np.array(records["static"].time)

    # Top: the targets and where each controller kept the block.
    top.plot(time, records["static"].target, color=INK_SOFT, linewidth=1.2, drawstyle="steps-post", label="target")
    for name in ("static", "oracle"):
        top.plot(time, records[name].position, color=COLOUR[name], linewidth=1.6, label=name)
    top.set_ylabel("position along the track (m)")
    top.legend(loc="upper left", frameon=False, ncols=3, labelcolor=INK_SOFT, bbox_to_anchor=(0.0, 1.13))
    top.annotate("until the first change, static\nand oracle match exactly\n(orange hides blue)",
                 xy=(1.0, 0.47), color=INK_SOFT, fontsize=8, va="center")

    # Middle and bottom: the hidden knobs, which neither controller is told (the oracle reads them).
    middle.plot(time, records["static"].friction, color=INK_SOFT, linewidth=1.6)
    middle.set_ylabel("friction\n(hidden)")
    middle.set_ylim(0.0, 0.75)
    bottom.plot(time, records["static"].mass, color=INK_SOFT, linewidth=1.6)
    bottom.set_ylabel("mass, kg\n(hidden)")
    bottom.set_ylim(0.8, 1.7)
    bottom.set_xlabel("time (s)")
    bottom.set_xlim(0, schedule.length)

    # Mark each hidden change across all three panels: a line for a jump, a light band for a drift.
    for change, knob, axis in ([(c, "friction", middle) for c in schedule.friction_changes]
                               + [(c, "mass", bottom) for c in schedule.mass_changes]):
        for ax in (top, middle, bottom):
            if change.over > 0:
                ax.axvspan(change.at, change.at + change.over, color=MUTED, alpha=0.10, linewidth=0)
            else:
                ax.axvline(change.at, color=AXIS, linewidth=1.0)
        axis.annotate(f"{knob} → {change.to:g}", xy=(change.at + change.over, axis.get_ylim()[1]),
                      xytext=(4, -12), textcoords="offset points", color=INK_SOFT, fontsize=8.5)

    for ax in (top, middle, bottom):
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

    fig.suptitle(f"The stream: static and oracle over {schedule.length:.0f} s", x=0.065, ha="left", color=INK, fontsize=13)
    fig.text(0.065, 0.925, "A new target every 6 s. Friction and mass change on a hidden schedule; "
             "neither controller is told, but the oracle reads the true values.",
             ha="left", color=INK_SOFT, fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.08, hspace=0.25)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
