"""Learning friction from free slides: the adaptive controller shoves the block to targets in turn while the friction
jumps partway through, unannounced.

    uv run python playground/21_learning_friction.py

Every 50 ms of free sliding gives a friction reading (the block's deceleration / g), and a running average that
forgets keeps the estimate the shove rule plans with. Prints the estimate before and after the jump, then draws the
readings, the estimate and the true friction over time, saves the picture to outputs/learning_friction.png and opens
it in a window (add --no-show to skip that).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary_free.arm_world import ArmWorld
from boundary_free.sensing import BlockSensor
from boundary_free.shove_controllers import AdaptiveShover

# ---- Things to change and re-run ---------------------------------------------
FRICTION_BEFORE, FRICTION_AFTER = 0.35, 0.5  # the true friction, before and after the jump
JUMP_AT = 60.0             # seconds; never on a new target (targets change every 6 s)
SPEED_NOISE = 0.01         # m/s of noise on each speed reading (stage F's model)
POSITION_NOISE = 0.002     # m of noise on each position reading
FORGETTING = 0.9           # per reading (about 3 per shove): a memory of about 1 / (1 - FORGETTING) readings
TARGET_COUNT, TARGET_EVERY = 20, 6.0
SEED = 0
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "learning_friction.png"
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ADAPTIVE, READING = "#1baf7a", "#9fdcc3"


def main():
    parser = argparse.ArgumentParser(description="The adaptive controller learning friction across a jump.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    targets = np.round(np.random.default_rng(SEED).uniform(-0.3, 0.3, TARGET_COUNT), 2)
    world = ArmWorld(friction=FRICTION_BEFORE)
    sensor = BlockSensor(world, position_noise=POSITION_NOISE, speed_noise=SPEED_NOISE, seed=SEED + 100)
    controller = AdaptiveShover(world, forgetting=FORGETTING, sensor=sensor)
    times, estimates, truths = [], [], []
    while world.time < TARGET_EVERY * TARGET_COUNT - 1e-9:
        if world.time >= JUMP_AT - 1e-9 and world.friction != FRICTION_AFTER:
            world.set_friction(FRICTION_AFTER)  # the hidden change: nothing tells the controller
        controller.update(targets[min(int(world.time // TARGET_EVERY), TARGET_COUNT - 1)])
        world.step()
        times.append(world.time)
        estimates.append(controller.estimator.friction)
        truths.append(world.friction)

    readings = np.array(controller.readings)
    print(f"friction {FRICTION_BEFORE} until {JUMP_AT:.0f} s, then {FRICTION_AFTER}; speed noise {1000 * SPEED_NOISE:g} "
          f"mm/s; forgetting {FORGETTING} (memory about {1 / (1 - FORGETTING):.0f} readings)")
    print(f"{len(readings)} readings from {len(controller.decisions)} shoves "
          f"({len(readings) / len(controller.decisions):.1f} per shove)")
    before = readings[(readings[:, 0] > 10) & (readings[:, 0] < JUMP_AT)]
    print(f"before the jump: readings {before[:, 1].mean():.3f} on average, each give or take {before[:, 1].std():.3f}; "
          f"estimate at {JUMP_AT:.0f} s: {estimates[np.searchsorted(times, JUMP_AT) - 1]:.3f}")
    after = [d for d in controller.decisions if d.time > JUMP_AT]
    caught_up = next((i + 1 for i, d in enumerate(after) if abs(d.friction_belief / FRICTION_AFTER - 1) < 0.05), None)
    print(f"after the jump: within 5% of {FRICTION_AFTER} by the {caught_up}th shove after it"
          if caught_up else f"after the jump: never within 5% of {FRICTION_AFTER}")
    draw(times, estimates, truths, readings)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def draw(times, estimates, truths, readings):
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    ax.scatter(readings[:, 0], readings[:, 1], s=10, color=READING, linewidths=0, label="readings (one per 50 ms of slide)")
    ax.plot(times, truths, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="true friction")
    ax.plot(times, estimates, color=ADAPTIVE, linewidth=1.8, label="estimate")
    ax.axvline(JUMP_AT, color=AXIS, linewidth=1)
    ax.annotate("friction jumps (unannounced)", xy=(JUMP_AT, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top", color=INK_SOFT, fontsize=8.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("friction coefficient")
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
    fig.suptitle("Learning friction from free slides", x=0.08, ha="left", color=INK, fontsize=12.5)
    fig.text(0.08, 0.915, f"Speed noise {1000 * SPEED_NOISE:g} mm/s; forgetting {FORGETTING} per reading. The estimate "
             "changes only while the block slides free.", ha="left", va="top", color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.11)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
