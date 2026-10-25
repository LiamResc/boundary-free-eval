"""The adaptive controller: it learns friction and mass from how the block responds, and what that buys it.

    uv run python playground/13_adaptive.py

Runs static, adaptive and oracle on the example stream and prints how each did, stretch by stretch. Then draws
the adaptive controller's beliefs against the truth, and static's and adaptive's regret, saves the picture to
outputs/adaptive.png and opens it in a window (add --no-show to skip that). Measuring regret twice takes ~30 s.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.controllers import OracleController, adaptive_controller, static_controller
from boundary_free.regret import measure_regret
from boundary_free.stream import Change, Schedule, Stream, example_schedule, run, targets_reached

# ---- Things to change and re-run ---------------------------------------------
# How much older sliding moments still count. Memory is about 1 / (1 - FORGETTING) sliding steps of 2 ms:
# 1 never forgets, 0.999 remembers about 2 s of sliding, 0.99 about 0.2 s.
FORGETTING = 0.99
# The hidden changes, written as in 11_the_stream.py. Neither static nor adaptive is told about them.
FRICTION_CHANGES = (Change(at=39, to=0.6), Change(at=81, to=0.3))
MASS_CHANGES = (Change(at=63, to=1.5, over=12),)
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "adaptive.png"
NAMES = ("static", "adaptive", "oracle")

# Colours: one per controller (adaptive is the palette's third), everything else in neutral greys.
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOUR = {"static": "#2a78d6", "oracle": "#eb6834", "adaptive": "#1baf7a"}


def main():
    parser = argparse.ArgumentParser(description="Run static, adaptive and oracle on the example stream.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()
    schedule = Schedule(targets=example_schedule().targets, target_every=6.0, friction_start=0.3, mass_start=1.0,
                        friction_changes=FRICTION_CHANGES, mass_changes=MASS_CHANGES)

    # 1. Run each controller through the whole stream, noting the adaptive controller's beliefs at every step.
    records, beliefs = {}, {"time": [], "friction": [], "mass": []}
    for name in NAMES:
        stream = Stream(schedule)
        controller = make(name, stream.world)
        note = None
        if name == "adaptive":
            def note(stream=stream, controller=controller):
                beliefs["time"].append(stream.time)
                beliefs["friction"].append(controller.friction_belief)
                beliefs["mass"].append(controller.mass_belief)
        records[name] = run(stream, controller, schedule.length, note)

    # 2. Measure static's and adaptive's regret against the oracle.
    print("Measuring regret for static and adaptive (about 30 s) ...")
    regret = {name: measure_regret(schedule, lambda world, name=name: make(name, world)) for name in ("static", "adaptive")}

    # 3. Print it all, then draw it.
    print_results(schedule, records, beliefs, regret)
    draw(schedule, records, beliefs, regret)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def make(name, world):
    """A fresh controller of the named kind. Only the oracle is allowed to read the world's friction and mass."""
    if name == "static":
        return static_controller()
    if name == "adaptive":
        return adaptive_controller(world, forgetting=FORGETTING)
    return OracleController(world)


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


def print_results(schedule, records, beliefs, regret):
    print(f"\nFORGETTING = {FORGETTING}: memory of about {1 / max(1 - FORGETTING, 1e-9):.0f} sliding steps")
    print("\nAverage miss in each stretch of the stream (cm):")
    print(f"{'':16s}{'':38s}" + "".join(f"{name:>9s}" for name in NAMES))
    for start, end in stretches(schedule):
        cells = []
        for name in NAMES:
            time = np.array(records[name].time)
            miss = np.abs(np.array(records[name].position) - np.array(records[name].target))
            cells.append(f"{100 * miss[(time > start) & (time <= end)].mean():9.2f}")
        print(f"{start:5.0f} to {end:3.0f} s   {describe(schedule, start, end):38s}{''.join(cells)}")

    print("\nTargets reached within 1 cm before the next one arrived (Y = reached, . = missed):")
    for name in NAMES:
        reached = targets_reached(records[name], schedule)
        print(f"   {name:>8}: {''.join('Y' if r else '.' for r in reached)}   {sum(reached)} of {len(reached)}")

    print("\nWhat adaptive believed at the end of each stretch (the truth in brackets):")
    time = np.array(beliefs["time"])
    for _, end in stretches(schedule):
        i = np.argmin(np.abs(time - (end - 0.002)))
        print(f"   at {end:3.0f} s: friction {beliefs['friction'][i]:.3f} ({schedule.friction(end - 0.002):.2f}),"
              f" mass {beliefs['mass'][i]:.3f} ({schedule.mass(end - 0.002):.2f})")

    print("\nRegret against the oracle, average in each stretch by checkpoint time (cm):")
    print(f"{'':16s}{'':38s}" + "".join(f"{name:>9s}" for name in regret))
    for start, end in stretches(schedule):
        cells = []
        for name in regret:
            times, values = np.array(regret[name].times), np.array(regret[name].regret)
            during = (times >= start) & (times < end)
            cells.append(f"{100 * values[during].mean():9.2f}" if during.any() else f"{'-':>9s}")
        print(f"{start:5.0f} to {end:3.0f} s   {describe(schedule, start, end):38s}{''.join(cells)}")
    print(f"{'whole stream (the rate)':54s}" + "".join(f"{100 * regret[name].rate:9.2f}" for name in regret))


def draw(schedule, records, beliefs, regret):
    fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5),
                                              gridspec_kw={"height_ratios": [1, 1, 1.4]}, facecolor=SURFACE)
    truth = records["adaptive"]

    # Top and middle: what adaptive believed, against the truth and against static's fixed beliefs.
    for ax, knob, believed_by_static, label in ((top, "friction", 0.3, "friction"), (middle, "mass", 1.0, "mass (kg)")):
        ax.plot(truth.time, getattr(truth, knob), color=INK_SOFT, linewidth=1.6, label="true value (hidden)")
        ax.axhline(believed_by_static, color=COLOUR["static"], linewidth=1.2, label="static believes")
        ax.plot(beliefs["time"], beliefs[knob], color=COLOUR["adaptive"], linewidth=1.6, label="adaptive believes")
        ax.set_ylabel(label)
    top.set_ylim(0.0, 0.9)
    middle.set_ylim(0.6, 2.3)
    top.legend(loc="lower left", frameon=False, ncols=3, labelcolor=INK_SOFT, fontsize=9, bbox_to_anchor=(0.0, 1.0))

    # Bottom: regret against the oracle at each checkpoint.
    bottom.axhline(0.0, color=AXIS, linewidth=1.0)
    for name in ("static", "adaptive"):
        times, values = np.array(regret[name].times), 100 * np.array(regret[name].regret)
        bottom.plot(times, values, color=COLOUR[name], linewidth=1.6, label=name)
        bottom.annotate(f"{name}: {100 * regret[name].rate:.2f} cm on average", xy=(times[-1], values[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK_SOFT, fontsize=8.5)
    bottom.set_ylabel("regret (cm)")
    bottom.set_xlabel("time (s)")
    bottom.set_xlim(0, schedule.length)
    bottom.legend(loc="upper left", frameon=False, labelcolor=INK_SOFT, fontsize=9)

    # Mark each hidden change on all three panels: a line for a jump, a light band for a drift.
    for knob, change, axis in ([("friction", c, top) for c in schedule.friction_changes]
                               + [("mass", c, middle) for c in schedule.mass_changes]):
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

    fig.suptitle("The adaptive controller learns friction and mass as it goes", x=0.065, ha="left", color=INK, fontsize=13)
    fig.text(0.065, 0.93, f"Its beliefs against the hidden truth, and its regret beside static's. FORGETTING = {FORGETTING}; "
             "no sensor noise.", ha="left", color=INK_SOFT, fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.83, top=0.86, bottom=0.07, hspace=0.3)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
