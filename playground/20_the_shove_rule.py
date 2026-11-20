"""The shove rule at work: static and the oracle shove the block to a few targets in turn.

    uv run python playground/20_the_shove_rule.py

Both use the same rule; static plans with its fixed friction belief and the oracle with the true friction. For each
target it prints how many shoves each needed and how long each took to get within 1 cm, then draws the block's
position over time against the targets, saves the picture to outputs/shove_rule.png and opens it in a window (add
--no-show to skip that).
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.arm_world import ArmWorld
from boundary_free.shove_controllers import TOLERANCE, OracleShover, ShoveController

# ---- Things to change and re-run ---------------------------------------------
TRUE_FRICTION = 0.45                    # the block's real friction coefficient
STATIC_BELIEF = 0.3                     # what static believes it is (the nominal value)
TARGETS = (0.2, -0.1, 0.25, 0.0, -0.3)  # where the block is asked to go, in turn, metres along the track
TARGET_EVERY = 6.0                      # seconds each target lasts
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "shove_rule.png"
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATIC, ORACLE = "#2a78d6", "#eb6834"


def target_at(t):
    return TARGETS[min(int(t // TARGET_EVERY), len(TARGETS) - 1)]


def run(make_controller):
    """Work through the targets; return the times, the block's positions and the controller."""
    world = ArmWorld(friction=TRUE_FRICTION)
    controller = make_controller(world)
    times, positions = [], []
    while world.time < TARGET_EVERY * len(TARGETS) - 1e-9:
        controller.update(target_at(world.time))
        world.step()
        times.append(world.time)
        positions.append(world.position)
    return np.array(times), np.array(positions), controller


def main():
    parser = argparse.ArgumentParser(description="Static and the oracle shove the block to a few targets.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    started = time.time()
    runs = {"static": run(lambda w: ShoveController(w, friction_belief=STATIC_BELIEF)),
            "oracle": run(OracleShover)}
    print(f"true friction {TRUE_FRICTION}; static believes {STATIC_BELIEF}; a new target every {TARGET_EVERY} s "
          f"({time.time() - started:.1f} s of computing for {2 * TARGET_EVERY * len(TARGETS):.0f} s simulated)\n")
    print("  target       static: shoves, within 1 cm after      oracle: shoves, within 1 cm after")
    for i, target in enumerate(TARGETS):
        cells = []
        for name in ("static", "oracle"):
            times, positions, controller = runs[name]
            start, end = i * TARGET_EVERY, (i + 1) * TARGET_EVERY
            shoves = sum(start <= d.time < end for d in controller.decisions)
            inside = (times >= start) & (times < end)
            near = np.nonzero(np.abs(positions[inside] - target) <= TOLERANCE)[0]
            reached = f"{times[inside][near[0]] - start:4.2f} s" if len(near) else "   never"
            cells.append(f"{shoves:3d}      {reached}")
        print(f"  {target:+.2f} m        {cells[0]}                   {cells[1]}")
    # Not printed: each run's average distance from the target. Over whole runs it is dominated by getting ready and
    # by where the block happens to wait for the next target, not by the friction estimate, so it cannot compare
    # controllers fairly; step 6's deviation regret compares them from the same situation instead.
    draw(runs)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def draw(runs):
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    t = runs["oracle"][0]
    ax.step(t, [target_at(x - 1e-9) for x in t], where="post", color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="target")
    for name, colour in (("static", STATIC), ("oracle", ORACLE)):
        times, positions, controller = runs[name]
        ax.plot(times, positions, color=colour, linewidth=1.8, label=name)
        decided = [d.time for d in controller.decisions]
        ax.plot(decided, [controller.world.position * 0 + positions[np.searchsorted(times, d)] for d in decided],
                linestyle="none", marker="|", markersize=10, color=colour)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("block's position along the track (m)")
    ax.legend(loc="upper right", frameon=False, labelcolor=INK_SOFT, fontsize=9)
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
    fig.suptitle("Same shove rule, different friction estimates", x=0.08, ha="left", color=INK, fontsize=12.5)
    fig.text(0.08, 0.915, f"True friction {TRUE_FRICTION}; static plans with {STATIC_BELIEF}, the oracle with the "
             "truth. Ticks mark each decision to shove.", ha="left", va="top", color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.11)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
