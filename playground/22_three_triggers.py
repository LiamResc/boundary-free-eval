"""The three triggers: the same adaptive controller, told about the change, detecting it itself, or only forgetting.

    uv run python playground/22_three_triggers.py

All three see the same targets and the same sensor noise, and friction changes once, at CHANGE_AT. They share one
response to a change (become unsure); only what triggers it differs. Prints, for each, how far its estimate was from
the truth after the change and how many shoves it took to get within 5%, then draws the three estimates against the
truth, saves the picture to outputs/three_triggers.png and opens it in a window (add --no-show to skip that).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.arm_world import ArmWorld
from boundary_free.sensing import BlockSensor
from boundary_free.shove_controllers import TRIGGERS, AdaptiveShover

# ---- Things to change and re-run ---------------------------------------------
FRICTION_BEFORE, FRICTION_AFTER = 0.35, 0.5  # try 0.385 for a small change: told pulls ahead of self-detected
CHANGE_AT = 63.0          # seconds; never on a new target (targets change every 6 s)
SPEED_NOISE = 0.01        # m/s of noise on each speed reading
FORGETTING = 0.9          # per reading; try 0.97 (a longer memory) and watch forgetting only fall behind
TARGET_COUNT, TARGET_EVERY = 20, 6.0
SEED = 0
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "three_triggers.png"
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOURS = {"told": "#2a78d6", "self-detected": "#1baf7a", "forgetting only": "#eb6834"}


def run(trigger, targets):
    world = ArmWorld(friction=FRICTION_BEFORE)
    sensor = BlockSensor(world, speed_noise=SPEED_NOISE, seed=SEED + 100)  # the same noise for all three
    controller = AdaptiveShover(world, trigger=trigger, forgetting=FORGETTING, sensor=sensor)
    times, estimates, truths = [], [], []
    while world.time < TARGET_EVERY * TARGET_COUNT - 1e-9:
        if world.time >= CHANGE_AT - 1e-9 and world.friction == FRICTION_BEFORE:
            world.set_friction(FRICTION_AFTER)  # the change; only the benchmark's announcement says so
            controller.told_of_change()         # heard only by the told version
        controller.update(targets[min(int(world.time // TARGET_EVERY), TARGET_COUNT - 1)])
        world.step()
        times.append(world.time)
        estimates.append(controller.estimator.friction)
        truths.append(world.friction)
    return np.array(times), np.array(estimates), np.array(truths), controller


def main():
    parser = argparse.ArgumentParser(description="The same adaptive controller with three triggers.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    targets = np.round(np.random.default_rng(SEED).uniform(-0.3, 0.3, TARGET_COUNT), 2)
    runs = {trigger: run(trigger, targets) for trigger in TRIGGERS}
    print(f"friction {FRICTION_BEFORE} -> {FRICTION_AFTER} at {CHANGE_AT:.0f} s; speed noise {1000 * SPEED_NOISE:g} mm/s; "
          f"forgetting {FORGETTING}\n")
    print("  trigger            estimate's error after the change   shoves to within 5%   alarms before / after")
    for trigger, (times, estimates, truths, controller) in runs.items():
        after = times > CHANGE_AT
        error = np.mean(np.abs(estimates[after] / FRICTION_AFTER - 1))
        later = [d for d in controller.decisions if d.time > CHANGE_AT]
        caught = next((str(i + 1) for i, d in enumerate(later) if abs(d.friction_belief / FRICTION_AFTER - 1) < 0.05),
                      "never")
        alarms = [t for t, cause in controller.resets if cause == "alarm"]
        alarm_text = (f"{sum(t < CHANGE_AT for t in alarms)} / {sum(t >= CHANGE_AT for t in alarms)}"
                      if trigger == "self-detected" else "-")
        print(f"  {trigger:<17s}        {100 * error:5.1f}%                          {caught:>5s}              "
              f"{alarm_text}")
    draw(runs)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def draw(runs):
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    times, _, truths, _ = runs["told"]
    ax.plot(times, truths, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="true friction")
    for trigger, (times, estimates, _, controller) in runs.items():
        wide = trigger == "told"  # drawn wide and pale underneath, so it still shows where self-detected matches it
        ax.plot(times, estimates, color=COLOURS[trigger], linewidth=4.5 if wide else 1.8, alpha=0.45 if wide else 1.0,
                label=trigger, zorder=1 if wide else 2)
        alarms = [t for t, cause in controller.resets if cause == "alarm"]
        ax.plot(alarms, [estimates[np.searchsorted(times, t)] for t in alarms], linestyle="none", marker="v",
                markersize=7, color=COLOURS[trigger])
    ax.axvline(CHANGE_AT, color=AXIS, linewidth=1)
    ax.annotate("friction changes (announced only to told)", xy=(CHANGE_AT, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", ha="left", va="top", color=INK_SOFT, fontsize=8.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("friction estimate")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_SOFT, fontsize=9)
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
    fig.suptitle("One response, three triggers", x=0.08, ha="left", color=INK, fontsize=12.5)
    fig.text(0.08, 0.915, f"Friction {FRICTION_BEFORE} -> {FRICTION_AFTER}; speed noise {1000 * SPEED_NOISE:g} mm/s; "
             f"forgetting {FORGETTING}. Triangles mark the self-detected version's alarms.\nBefore the change all three are one "
             "line (same noise, same targets); after it, told (wide, pale) and self-detected can coincide.",
             ha="left", va="top", linespacing=1.4,
             color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.11)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
