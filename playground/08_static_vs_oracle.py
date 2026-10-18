"""Static versus oracle: the same pushing controller, with assumed numbers or with the true ones.

Numbers for both, side by side:
    uv run python playground/08_static_vs_oracle.py --headless
Watch one of them in a window (set WATCH below; on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/08_static_vs_oracle.py
In the window: red arrow = the pusher, blue arrow = friction (4 cm per newton), green post = the target.
"""
import argparse

import mujoco.viewer
import numpy as np

from boundary_free.block_world import BlockWorld
from boundary_free.controllers import OracleController, static_controller
from boundary_free.simulate import drive
from boundary_free.viewer_tools import RealTime, draw_forces

# ---- Things to change and re-run ---------------------------------------------
TRUE_FRICTION = 0.6  # the block's real friction (static always believes 0.3)
TRUE_MASS = 1.0      # the block's real mass, in kilograms (static always believes 1.0)
STIFFNESS = 25.0     # the controllers' stiffness (braking is matched to it, so there is no overshoot)
WATCH = "static"     # which controller the window shows: "static" or "oracle"
# -------------------------------------------------------------------------------

START, TARGET, SECONDS = -0.3, 0.3, 4.0


def make(name):
    """A fresh world with the true friction and mass, the block at the start, and the named controller."""
    world = BlockWorld(friction=TRUE_FRICTION, mass=TRUE_MASS)
    world.data.qpos[0] = START
    controller = static_controller(stiffness=STIFFNESS) if name == "static" else OracleController(world, stiffness=STIFFNESS)
    return world, controller


def summary(name, record):
    """One line on how it went: where it ended, when it got within 1 cm, and its average miss."""
    misses = np.abs(TARGET - np.array(record.position))
    close = next((t for t, m in zip(record.time, misses) if m < 0.01), None)
    when = f"within 1 cm after {close:.2f} s" if close is not None else "never within 1 cm"
    return (f"{name:>6}: ends {100 * (TARGET - record.position[-1]):+6.2f} cm from the target, {when}, "
            f"average miss {100 * misses.mean():5.2f} cm")


def side_by_side():
    """Run both controllers on identical worlds and print what each did, every quarter second."""
    records = {}
    for name in ("static", "oracle"):
        world, controller = make(name)
        records[name] = drive(world, controller, TARGET, SECONDS)

    print(f"true friction {TRUE_FRICTION}, true mass {TRUE_MASS} kg; static believes friction 0.3, mass 1.0 kg")
    print(f"block starts at {START} m, target {TARGET} m, stiffness {STIFFNESS}")
    print()
    print("          ------ static ------      ------ oracle ------")
    print("  time    miss (cm)   push (N)      miss (cm)   push (N)")
    every = round(0.25 / BlockWorld().timestep)
    for i in range(every - 1, len(records["static"].time), every):
        s, o = records["static"], records["oracle"]
        print(f"{s.time[i]:6.2f}    {100 * (TARGET - s.position[i]):+8.2f}   {s.push[i]:+8.2f}      "
              f"{100 * (TARGET - o.position[i]):+8.2f}   {o.push[i]:+8.2f}")
    print()
    for name in ("static", "oracle"):
        print(summary(name, records[name]))


def watch():
    """Show one controller in a window, then keep it holding the target until the window is closed."""
    world, controller = make(WATCH)
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        pace = RealTime(world)

        def show():
            draw_forces(viewer, world, target=TARGET)
            viewer.sync()
            pace()

        record = drive(world, controller, TARGET, SECONDS, on_step=show)
        print(summary(WATCH, record))
        while viewer.is_running():
            drive(world, controller, TARGET, 0.1, on_step=show)


def main():
    parser = argparse.ArgumentParser(description="Compare the static and oracle pushing controllers.")
    parser.add_argument("--headless", action="store_true", help="no window: print both side by side")
    args = parser.parse_args()
    side_by_side() if args.headless else watch()


if __name__ == "__main__":
    main()
