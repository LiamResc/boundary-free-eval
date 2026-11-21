"""The shove: the ball carries the block up to the release speed, brakes hard, and the block slides on alone.

    uv run python playground/19_the_shove.py

Gets the ball ready beside the block, shoves it once, and waits for it to stop. Prints the plan (carried, slide,
total) against what happened, the free slide's deceleration against friction x g, and the arm's joint speeds and
accelerations against the real FR3's limits. Then draws the ball's and the block's speed along the track over time,
saves the picture to outputs/shove.png and opens it in a window (add --no-show to skip that).

To watch it in the simulator's window instead (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/19_the_shove.py --watch
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

from boundary_free.arm_world import TRACK_AXIS, ArmWorld
from boundary_free.block_world import GRAVITY
from boundary_free.motion import (A_BRAKE, A_PUSH, STANDOFF, JOINT_SPEED_LIMITS, PusherMotion, Shove, block_travel,
                                  trajectory_to_ready)
from boundary_free.viewer_tools import RealTime

# ---- Things to change and re-run ---------------------------------------------
FRICTION = 0.01      # the block's friction coefficient
RELEASE_SPEED = 0.6  # how fast the ball is going when it starts braking, m/s
BRAKE = A_BRAKE      # how hard the ball brakes, m/s^2 (8). Try 3 with FRICTION = 0.4: below friction x g, no release
DIRECTION = +1       # +1 shoves towards +y (the robot's left), -1 towards -y
# -------------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "shove.png"
JOINT_ACCELERATION_LIMIT = 10.0  # rad/s^2, every joint of the real FR3

SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BALL, BLOCK = "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description="Shove the block once and compare with the formula.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    parser.add_argument("--watch", action="store_true", help="watch in the simulator's window (needs mjpython)")
    args = parser.parse_args()

    world = ArmWorld(friction=FRICTION)
    if args.watch:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            record = run(world, viewer)
    else:
        record = run(world)
    report(record)
    draw(record)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show and not args.watch:
        plt.show()


def run(world, viewer=None):
    """Get ready, shove, and wait for the block to stop. Returns what happened during the shove, step by step."""
    motion = PusherMotion(world)
    pace = RealTime(world)

    def steer(seconds, log=None):
        end = world.time + seconds
        while world.time < end - 1e-9:
            motion.update()
            before = world.tip_position
            world.step()
            if log is not None:
                ball_speed = np.dot(world.tip_position - before, TRACK_AXIS) / world.timestep * DIRECTION
                log.append((world.time, ball_speed, world.velocity * DIRECTION, world.position,
                            world.pusher_touches_block, world.joint_velocities,
                            np.dot(motion.trajectory.reference(motion.elapsed)[1], TRACK_AXIS)
                            * DIRECTION))
            if viewer is not None:
                viewer.sync()
                pace()

    # 1. Get ready: the ball beside the block, a STANDOFF from the face it will push.
    motion.follow(trajectory_to_ready(world.tip_position, world.position, DIRECTION))
    steer(motion.trajectory.duration + 0.3)

    # 2. Shove, and wait for the block to stop.
    shove = Shove(world.tip_position, DIRECTION, RELEASE_SPEED, brake_acceleration=BRAKE)
    start, log = world.position, []
    motion.follow(shove)
    steer(shove.duration + 1.2, log)
    t0 = log[0][0] - world.timestep
    columns = list(zip(*log))
    return dict(shove=shove, start=start, time=np.array(columns[0]) - t0, ball=np.array(columns[1]),
                block=np.array(columns[2]), position=np.array(columns[3]), touching=np.array(columns[4]),
                joints=np.array(columns[5]), reference=np.array(columns[6]))


def report(r):
    shove, v = r["shove"], RELEASE_SPEED
    last = np.nonzero(r["touching"])[0].max()                  # the last step the ball touched the block
    released_at = (r["position"][last + 1] - r["start"]) * DIRECTION
    release_speed = r["block"][last + 1]
    total = (r["position"][-1] - r["start"]) * DIRECTION
    carried_plan = v**2 / (2 * A_PUSH) - STANDOFF
    slide_plan = v**2 / (2 * FRICTION * GRAVITY)
    print(f"friction {FRICTION}, release speed {v} m/s, speeding up at {A_PUSH} and braking at {BRAKE} m/s^2")
    print(f"friction x g = {FRICTION * GRAVITY:.2f} m/s^2: braking {'beats' if BRAKE > FRICTION * GRAVITY else 'does NOT beat'} it\n")
    print("                      plan       happened")
    print(f"  carried          {100 * carried_plan:6.1f} cm   {100 * released_at:6.1f} cm   (to the last moment of contact)")
    print(f"  slide            {100 * slide_plan:6.1f} cm   {100 * (total - released_at):6.1f} cm")
    print(f"  total            {100 * block_travel(v, FRICTION):6.1f} cm   {100 * total:6.1f} cm")
    print(f"  release speed    {v:6.3f} m/s {release_speed:6.3f} m/s")
    sliding = (np.arange(len(r["time"])) > last + 5) & (r["block"] > 0.05)
    if sliding.sum() > 10:
        slope = np.polyfit(r["time"][sliding], r["block"][sliding], 1)[0]
        print(f"\nthe free slide slowed at {-slope:.3f} m/s^2; friction x g = {FRICTION * GRAVITY:.3f}")
    else:
        print("\nno free slide: the block was carried to a stop")
    accelerations = np.abs(r["joints"][5:] - r["joints"][:-5]) / (5 * 0.002)
    print(f"fastest joint {100 * (np.abs(r['joints']) / JOINT_SPEED_LIMITS).max():.0f}% of its FR3 limit; "
          f"largest joint acceleration (10 ms average) {accelerations.max():.1f} rad/s^2 "
          f"(limit {JOINT_ACCELERATION_LIMIT:.0f})")


def draw(r):
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    t = r["time"]
    ax.plot(t, r["reference"], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="ball's reference")
    ax.plot(t, r["ball"], color=BALL, linewidth=1.8, label="ball")
    ax.plot(t, r["block"], color=BLOCK, linewidth=1.8, label="block")
    last = np.nonzero(r["touching"])[0].max()
    ax.axvline(t[last + 1], color=AXIS, linewidth=1)
    ax.annotate("released", xy=(t[last + 1], 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top", color=INK_SOFT, fontsize=8.5)
    ax.annotate(f"slides on, slowing at friction x g = {FRICTION * GRAVITY:.2f} m/s²",
                xy=(t[last + 1] + 0.05, max(r["block"][last + 1] - FRICTION * GRAVITY * 0.05, 0.02)), xytext=(14, 12),
                textcoords="offset points", ha="left", va="bottom", color=INK_SOFT, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
    ax.set_xlabel("time since the shove began (s)")
    ax.set_ylabel("speed along the track (m/s)")
    ax.set_ylim(bottom=min(0.0, r["ball"].min()) - 0.02)
    moving = np.nonzero((np.abs(r["block"]) > 1e-3) | (np.abs(r["ball"]) > 1e-2))[0]
    ax.set_xlim(0, t[moving.max()] + 0.1)  # stop the picture shortly after everything has come to rest
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
    fig.suptitle("The ball carries the block up to speed, brakes, and the block slides on alone", x=0.08,
                 ha="left", color=INK, fontsize=12.5)
    total = (r["position"][-1] - r["start"]) * DIRECTION
    fig.text(0.08, 0.915, f"Friction {FRICTION}, release speed {RELEASE_SPEED} m/s, braking {BRAKE} m/s². The block "
             f"travelled {100 * total:.1f} cm; the formula says {100 * block_travel(RELEASE_SPEED, FRICTION):.1f} cm.",
             ha="left", va="top", color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.11)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
