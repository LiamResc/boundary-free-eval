"""Push the block: set its friction, its mass and a push, then compare what it does with the equation.

Numbers only:
    uv run python playground/07_push_the_block.py --headless
With a window (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/07_push_the_block.py
"""
import argparse
import time

import mujoco.viewer

from boundary_free.block_world import GRAVITY, BlockWorld

# ---- Things to change and re-run ---------------------------------------------
FRICTION = 0.3   # friction coefficient between block and table
MASS = 1.0       # the block's mass, in kilograms
PUSH = 6.0       # the push, in newtons (the pusher can give at most 20)
PUSH_TIME = 0.5  # how long to push, in seconds, before the pusher lets go
# -------------------------------------------------------------------------------


def run(world, viewer=None):
    """Push for PUSH_TIME seconds, let go, and report each phase."""
    limit = world.friction_limit
    print(f"friction limit = friction x mass x g = {FRICTION} x {MASS} x {GRAVITY} = {limit:.2f} N")
    push = max(-world.push_limit, min(world.push_limit, PUSH))
    if push != PUSH:
        print(f"note: the pusher can give at most {world.push_limit:.0f} N, so it pushes with {push:.0f} N")

    # 1. Push for PUSH_TIME seconds.
    world.push(PUSH)
    reached_end = False
    while world.time < PUSH_TIME - 1e-9:
        world.step()
        reached_end = reached_end or abs(world.position) > 0.99
        show(world, viewer)
    released_at, speed = world.position, world.velocity

    print()
    if reached_end:
        print("the block reached the end of the track while being pushed, and the end stopped it.")
        print("Try a smaller PUSH or a shorter PUSH_TIME to compare with the equation.")
        keep_window_open(world, viewer)
        return
    if abs(push) <= limit:
        print(f"pushing with {push:.2f} N: the push is under the limit, so the block sticks "
              f"(it crept {1000 * abs(released_at):.3f} mm)")
    else:
        predicted = (abs(push) - limit) / MASS
        print(f"pushing with {push:.2f} N for {PUSH_TIME} s: the block slides")
        print(f"   acceleration: measured {abs(speed) / PUSH_TIME:.3f} m/s^2, "
              f"equation says ({abs(push):.2f} - {limit:.2f}) / {MASS} = {predicted:.3f}")
        print(f"   speed when the pusher lets go: {abs(speed):.3f} m/s")

    # 2. Let go, and let friction bring the block to a stop.
    world.push(0.0)
    while abs(world.velocity) > 1e-4 and world.time < PUSH_TIME + 10.0:
        world.step()
        reached_end = reached_end or abs(world.position) > 0.99
        show(world, viewer)
    if reached_end:
        print("coasting: the block reached the end of the track, which stopped it early")
    elif abs(push) > limit:
        print(f"coasting: it slid {abs(world.position - released_at):.4f} m before stopping, "
              f"equation says v^2 / (2 x friction x g) = {speed ** 2 / (2 * FRICTION * GRAVITY):.4f} m")
    keep_window_open(world, viewer)


def keep_window_open(world, viewer):
    """With a window open, keep simulating until it is closed."""
    while viewer is not None and viewer.is_running():
        world.step()
        show(world, viewer)


def show(world, viewer):
    """With a window open: draw the new state, then wait so the motion plays at real speed."""
    if viewer is None:
        return
    viewer.sync()
    ahead = world.time - (time.time() - show.wall_start)
    if ahead > 0:
        time.sleep(ahead)


def main():
    parser = argparse.ArgumentParser(description="Push the block and compare with the equation.")
    parser.add_argument("--headless", action="store_true", help="no window: print the numbers only")
    args = parser.parse_args()

    world = BlockWorld(friction=FRICTION, mass=MASS)
    show.wall_start = time.time()
    if args.headless:
        run(world)
    else:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            show.wall_start = time.time()
            run(world, viewer)


if __name__ == "__main__":
    main()
