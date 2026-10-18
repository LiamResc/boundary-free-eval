"""What wrong beliefs cost: static versus oracle across many true frictions and masses.

    uv run python playground/09_cost_of_wrong_beliefs.py

For each combination of true friction and true mass, both controllers push the block from -0.3 m to a
target at +0.3 m for 4 seconds. The table shows how much bigger static's average miss is than the
oracle's, in centimetres. Static believes friction 0.3 and mass 1.0 kg, so that cell should be 0.
"""
import numpy as np

from boundary_free.block_world import BlockWorld
from boundary_free.controllers import OracleController, static_controller
from boundary_free.simulate import drive

# ---- Things to change and re-run ---------------------------------------------
STIFFNESS = 25.0
FRICTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MASSES = [0.5, 1.0, 1.5, 2.0]
# -------------------------------------------------------------------------------

START, TARGET, SECONDS = -0.3, 0.3, 4.0


def average_miss(friction, mass, oracle):
    """Run one controller on one world and return its average distance from the target, in metres."""
    world = BlockWorld(friction=friction, mass=mass)
    world.data.qpos[0] = START
    controller = OracleController(world, stiffness=STIFFNESS) if oracle else static_controller(stiffness=STIFFNESS)
    record = drive(world, controller, TARGET, SECONDS)
    return float(np.mean(np.abs(TARGET - np.array(record.position))))


def main():
    print(f"Extra average miss of static over the oracle, in cm (stiffness {STIFFNESS})")
    print()
    print("true friction  " + "".join(f"  mass {m:3.1f}" for m in MASSES))
    for friction in FRICTIONS:
        cells = []
        for mass in MASSES:
            extra = average_miss(friction, mass, oracle=False) - average_miss(friction, mass, oracle=True)
            cells.append(f"{100 * extra:10.2f}")
        print(f"     {friction:4.2f}     " + "".join(cells))


if __name__ == "__main__":
    main()
