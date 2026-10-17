"""Push the block: set its friction, its mass and a push, then compare what it does with the equation.

Numbers only:
    uv run python playground/07_push_the_block.py --headless
With a window (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/07_push_the_block.py
In the window, a red arrow shows the pusher's force and a blue arrow shows friction (4 cm per newton).
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from boundary_free.block_world import GRAVITY, BlockWorld

BLOCK_HALF = 0.05    # the block is a 10 cm cube
BLOCK_HEIGHT = 0.8   # height of the block's centre above the floor, in metres
TABLE_TOP = 0.75     # height of the table top, in metres
ARROW_SCALE = 0.04   # metres of arrow per newton of force

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
    draw_forces(world, viewer)
    viewer.sync()
    ahead = world.time - (time.time() - show.wall_start)
    if ahead > 0:
        time.sleep(ahead)


def draw_forces(world, viewer):
    """Draw the two forces on the block as arrows in the window.

    Neither force comes from anything visible: the pusher is an invisible motor on the block's track,
    and friction is set on the track too. The arrows make them visible.
    """
    scene = viewer.user_scn
    scene.ngeom = 0
    x = world.position

    # Red: the pusher's force, drawn as a push arriving at the back face of the block.
    push = world.applied_push
    if abs(push) > 0.01:
        direction = np.sign(push)
        tip = np.array([x - direction * BLOCK_HALF, 0.0, BLOCK_HEIGHT])
        tail = tip - np.array([direction * ARROW_SCALE * abs(push), 0.0, 0.0])
        add_arrow(scene, tail, tip, [0.9, 0.1, 0.1, 1.0])

    # Blue: friction, drawn beside the block at the table top, where the rubbing happens.
    friction = world.friction_force
    if abs(friction) > 0.01:
        tail = np.array([x, -(BLOCK_HALF + 0.05), TABLE_TOP + 0.01])
        tip = tail + np.array([ARROW_SCALE * friction, 0.0, 0.0])
        add_arrow(scene, tail, tip, [0.1, 0.4, 0.9, 1.0])


def add_arrow(scene, tail, tip, rgba):
    """Add one arrow from tail to tip to the window's extra drawings."""
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, tail, tip)
    scene.ngeom += 1


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
