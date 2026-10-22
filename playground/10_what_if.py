"""What if? Branch the simulation at one moment and compare two futures that start from exactly the same place.

    uv run python playground/10_what_if.py

The main run uses the static controller on a block whose real friction is higher than static believes.
At BRANCH_AT seconds we take a branch: an exact copy of that moment. On the branch the oracle takes over;
in the main run static carries on. Both futures start from the same situation, so any difference between
them is down to the controllers alone.
"""
import numpy as np

from boundary_free.block_world import BlockWorld
from boundary_free.controllers import OracleController, static_controller
from boundary_free.simulate import drive

# ---- Things to change and re-run ---------------------------------------------
TRUE_FRICTION = 0.6  # static believes 0.3
TRUE_MASS = 1.0      # static believes 1.0
BRANCH_AT = 2.0      # seconds into the run when we take the branch
LOOK_AHEAD = 0.5     # how long each future runs, in seconds
# -------------------------------------------------------------------------------

START, TARGET = -0.3, 0.3


def main():
    # 1. The main run, with the static controller, up to the moment we branch.
    world = BlockWorld(friction=TRUE_FRICTION, mass=TRUE_MASS)
    world.data.qpos[0] = START
    static = static_controller()
    drive(world, static, TARGET, BRANCH_AT)
    print(f"at {world.time:.2f} s the block is {100 * (TARGET - world.position):.2f} cm short of the target, "
          f"moving at {100 * world.velocity:.2f} cm/s")

    # 2. Save the moment, and make a branch: an exact copy with its own model and data.
    saved = world.snapshot()
    twin = world.branch()

    # 3. Two futures from the same start: static carries on in the main run, the oracle takes over the branch.
    main_future = drive(world, static, TARGET, LOOK_AHEAD)
    branch_future = drive(twin, OracleController(twin), TARGET, LOOK_AHEAD)

    print()
    print("          main run: static     branch: oracle")
    print("  time       miss (cm)           miss (cm)")
    every = round(0.25 / world.timestep)
    for i in range(every - 1, len(main_future.time), every):
        print(f"{main_future.time[i]:6.2f}       {100 * (TARGET - main_future.position[i]):+7.2f}            "
              f"{100 * (TARGET - branch_future.position[i]):+7.2f}")

    static_miss = np.mean(np.abs(TARGET - np.array(main_future.position)))
    oracle_miss = np.mean(np.abs(TARGET - np.array(branch_future.position)))
    print()
    print(f"average miss over the {LOOK_AHEAD} s: static {100 * static_miss:.2f} cm, oracle {100 * oracle_miss:.2f} cm")
    print(f"from this moment, the oracle would have been {100 * (static_miss - oracle_miss):.2f} cm closer on average.")
    print("Measured at many moments, a difference like this is what stage E calls regret.")

    # 4. Check that the moment was saved exactly: replay the main future from the snapshot.
    world.restore(saved)
    replay = drive(world, static, TARGET, LOOK_AHEAD)
    print()
    print(f"replaying the main future from the snapshot gives the same result to the last digit: "
          f"{replay.position == main_future.position}")


if __name__ == "__main__":
    main()
