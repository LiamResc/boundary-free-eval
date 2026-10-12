"""Where is the hand? Joint angles in, hand position out.

Numbers only:
    uv run python playground/03_where_is_the_hand.py --headless
With a window, to watch the arm make the turn (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/03_where_is_the_hand.py
"""
import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"
HAND = "attachment_site"  # the point at the end of the arm's last link, where a tool gets mounted

# ---- Things to change and re-run ---------------------------------------------
JOINT = "fr3_joint7"  # which joint to turn: fr3_joint1 (at the base) to fr3_joint7 (at the wrist)
TURN = 0.5            # how far to turn it from its home angle, in radians (0.5 rad is about 29 degrees)
# -------------------------------------------------------------------------------


def load():
    """Build the model (what exists) and the data (the situation right now), at the home pose."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    return model, data


def hand_position(data):
    """Where the hand is: x (forward), y (to the robot's left), z (up), in metres from the centre of the base."""
    return data.site(HAND).xpos.copy()


def work_it_out(model, data):
    """By geometry: place the joint at the new angle and ask where the hand would be. No motors, no gravity, no time."""
    # 1. mj_forward works out where every part of the arm is from the joint angles held in the data.
    mujoco.mj_forward(model, data)
    at_home = hand_position(data)

    # 2. Change one joint angle in the data and work it out again.
    home_angle = data.joint(JOINT).qpos[0]
    low, high = model.joint(JOINT).range
    if not low <= home_angle + TURN <= high:
        print(f"note: {home_angle + TURN:+.3f} rad is outside this joint's range ({low:+.3f} to {high:+.3f})")
    data.joint(JOINT).qpos[0] = home_angle + TURN
    mujoco.mj_forward(model, data)
    after_turn = hand_position(data)

    # 3. Report both, and the difference.
    moved = after_turn - at_home
    print(f"turning {JOINT} by {TURN:+.2f} rad (from its home angle {home_angle:+.3f} to {home_angle + TURN:+.3f})")
    print()
    print("By geometry: the joint is simply placed at the new angle. No motors, no gravity, no time.")
    print("                         x (forward)  y (left)     z (up)     in metres")
    print(f"hand at home             {row(at_home)}")
    print(f"hand after the turn      {row(after_turn)}")
    print(f"change                   {row(moved)}    ({100 * math.sqrt(sum(moved ** 2)):.1f} cm in a straight line)")
    return at_home, after_turn


def row(xyz):
    """Three numbers lined up under the x, y, z headings (and no '-0.000')."""
    return "       ".join(f"{round(v, 3) + 0.0:+.3f}" for v in xyz)


def watch(model, data, viewer, after_turn):
    """By physics: tell the motor to go to the new angle, let the arm move there under gravity, and see where the hand ends up."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    motor = model.actuator(JOINT).id
    home_angle = data.ctrl[motor]
    reported = False
    wall_start = time.time()

    while viewer.is_running():
        # The command glides from the home angle to the new angle over two seconds, then stays there.
        data.ctrl[motor] = home_angle + TURN * min(data.time / 2.0, 1.0)
        mujoco.mj_step(model, data)

        # Two seconds after the command arrives, compare where the motors got the hand with the geometry.
        if not reported and data.time >= 4.0:
            reached = hand_position(data)
            off = 1000 * math.sqrt(sum((reached - after_turn) ** 2))
            print()
            print("By physics: the motor is told to go to the new angle, and the arm moves there under gravity.")
            print(f"hand after the turn      {row(reached)}    ({off:.1f} mm from the geometry's answer)")
            reported = True

        viewer.sync()
        ahead = data.time - (time.time() - wall_start)
        if ahead > 0:
            time.sleep(ahead)


def main():
    parser = argparse.ArgumentParser(description="Turn one FR3 joint and see where the hand goes.")
    parser.add_argument("--headless", action="store_true", help="no window: print the geometry only")
    args = parser.parse_args()

    model, data = load()
    at_home, after_turn = work_it_out(model, data)
    if not args.headless:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            watch(model, data, viewer, after_turn)


if __name__ == "__main__":
    main()
