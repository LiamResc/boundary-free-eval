"""Swing one joint of the FR3 back and forth from our own simulation loop.

With a window (on macOS this kind of viewer needs the `mjpython` launcher):
    uv run mjpython playground/01_move_a_joint.py
Without a window, printing numbers only:
    uv run python playground/01_move_a_joint.py --headless
"""
import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"

# ---- Things to change and re-run ---------------------------------------------
JOINT = "fr3_joint1"  # which joint to swing: fr3_joint1 (at the base) to fr3_joint7 (at the wrist)
SWING = 0.5           # how far either side of the home angle, in radians (0.5 rad is about 29 degrees)
PERIOD = 4        # seconds for one full back-and-forth
# -------------------------------------------------------------------------------


def load():
    """Build the model (what exists) and the data (the situation right now), at the home pose."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    return model, data


def run(model, data, viewer=None, seconds=None):
    """The simulation loop. Stops when the window is closed, or after `seconds` of simulated time."""
    # In this model each motor has the same name as the joint it drives.
    motor = model.actuator(JOINT).id
    home_angle = data.ctrl[motor]
    next_report = 0.0
    largest_gap = 0.0
    wall_start = time.time()

    while (viewer is None or viewer.is_running()) and (seconds is None or data.time <= seconds + 1e-9):
        # 1. Decide the command: the angle we want this joint's motor to aim for right now.
        target = home_angle + SWING * math.sin(2 * math.pi * data.time / PERIOD)
        data.ctrl[motor] = target

        # 2. Compare what we are asking for with where the joint is at this same instant.
        #    Print it eight times per swing, so the same eight moments show up whatever PERIOD is.
        actual = data.joint(JOINT).qpos[0]
        largest_gap = max(largest_gap, abs(target - actual))
        if data.time >= next_report - 1e-9:
            print(f"t={data.time:6.3f} s   commanded={target:+.3f}   actual={actual:+.3f}   gap={target - actual:+.4f} rad")
            next_report += PERIOD / 8

        # 3. Advance the physics by one timestep (2 milliseconds of simulated time).
        mujoco.mj_step(model, data)

        # 4. With a window open: draw the new state, then wait for the real clock to catch up
        #    with simulated time, so the motion plays at real speed.
        if viewer is not None:
            viewer.sync()
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

    print(f"largest gap: {largest_gap:.4f} rad ({math.degrees(largest_gap):.1f} degrees)")


def main():
    parser = argparse.ArgumentParser(description="Swing one FR3 joint back and forth.")
    parser.add_argument("--headless", action="store_true", help="no window: simulate two swings and print")
    args = parser.parse_args()

    model, data = load()
    if args.headless:
        run(model, data, seconds=2 * PERIOD)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(model, data, viewer=viewer)


if __name__ == "__main__":
    main()
