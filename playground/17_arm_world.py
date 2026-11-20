"""The arm world's scene: the FR3 with a ball-tipped pusher, and the block on its track across the table.

With a window:
    uv run python playground/17_arm_world.py
Numbers only:
    uv run python playground/17_arm_world.py --headless

Prints where things are and two checks: the arm holding its home pose, and the block responding to a push exactly as
in the block world. In the window the arm holds its home pose. Double-click the block to select it, then hold Ctrl
and drag with the right mouse button to push it: it only moves along its track, and it sticks until the push beats
its friction.
"""
import argparse

import mujoco.viewer
import numpy as np

from boundary_free.arm_world import TABLE_HEIGHT, TRACK_AXIS, TRACK_HALF_LENGTH, TRACK_X, ArmWorld

# ---- Things to change and re-run ---------------------------------------------
FRICTION = 0.3   # friction coefficient between block and table
MASS = 1.0       # the block's mass, in kilograms
TEST_PUSH = 5.0  # a steady push on the block along its track, in newtons, for the check
# -------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Look at the arm world's scene and check it.")
    parser.add_argument("--headless", action="store_true", help="print the checks without opening a window")
    args = parser.parse_args()

    describe(ArmWorld(friction=FRICTION, mass=MASS))
    check_holding(ArmWorld(friction=FRICTION, mass=MASS))
    check_push(ArmWorld(friction=FRICTION, mass=MASS))
    if not args.headless:
        world = ArmWorld(friction=FRICTION, mass=MASS)
        mujoco.viewer.launch(world.model, world.data)


def describe(world):
    """Where things are, in metres: x away from the robot, y to its left, z up from the table top."""
    tip, flange = world.tip_position, world.data.site("attachment_site").xpos
    print("The scene (metres; x away from the robot, y to its left, z up from the table top)")
    print(f"  table top level with the robot's base; floor {TABLE_HEIGHT} m below")
    print(f"  block: a 10 cm cube on a track along y at x = {TRACK_X}, free to slide {TRACK_HALF_LENGTH} m either way")
    print(f"  pusher's ball at home: ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}), "
          f"{flange[2] - tip[2]:.2f} m below the flange")
    print("  touching: arm and pusher with table and block; never block with table, nor arm with itself")


def check_holding(world):
    """Hold the home pose for two seconds: gravity compensation means no sag."""
    tip = world.tip_position
    for _ in range(int(2.0 / world.timestep)):
        world.step()
    error = np.abs(world.joint_angles - world.home).max()
    moved = np.linalg.norm(world.tip_position - tip)
    print(f"\nHolding home for 2 s: largest joint position error {1000 * error:.3f} mrad, "
          f"the ball moved {1000 * moved:.3f} mm")


def check_push(world):
    """Push the block steadily along its track for 0.3 s and compare with the block world's equation."""
    limit = world.friction_limit
    for _ in range(int(0.3 / world.timestep)):
        world.data.xfrc_applied[world.block_body, :3] = TEST_PUSH * TRACK_AXIS
        world.step()
    print(f"\nPushing the block with {TEST_PUSH} N along its track (friction limit {limit:.3f} N):")
    if TEST_PUSH <= limit:
        print(f"  it sticks: it moved {1000 * world.position:.3f} mm, friction pushing back {world.friction_force:.3f} N")
    else:
        expected = (TEST_PUSH - limit) / world.mass
        print(f"  acceleration {world.velocity / 0.3:.3f} m/s^2; the equation (push - friction limit) / mass "
              f"gives {expected:.3f}")
        print(f"  friction pushing back {world.friction_force:.3f} N; shapes touching: {world.data.ncon}")


if __name__ == "__main__":
    main()
