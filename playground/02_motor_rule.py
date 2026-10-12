"""The rule a position motor follows: a spring toward the commanded angle, plus a damper.

Same swing as 01_move_a_joint.py, but here we set the motor's two numbers ourselves and
work out its push by hand, to see where the gap comes from.

Numbers only:
    uv run python playground/02_motor_rule.py --headless
With a window (on macOS this kind of viewer needs the `mjpython` launcher):
    uv run mjpython playground/02_motor_rule.py
"""
import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"
JOINT = "fr3_joint1"  # this script is about one motor: the one at the base of the arm

# ---- Things to change and re-run ---------------------------------------------
STIFFNESS = 4500.0  # the spring: newton-metres of push per radian of gap (4500 is the model's own value)
DAMPING = 450.0     # the damper: newton-metres of resistance per radian-per-second of speed (model's own: 450)
SWING = 0.5         # how far either side of the home angle, in radians
PERIOD = 4.0        # seconds for one full back-and-forth
# -------------------------------------------------------------------------------


def load():
    """Build the model and the data at the home pose, then give the joint's motor our two numbers."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # MuJoCo stores a motor's rule as a few numbers. With these three filled in, its push is
    #     STIFFNESS * (commanded angle - actual angle)  -  DAMPING * (joint speed)
    motor = model.actuator(JOINT).id
    model.actuator_gainprm[motor, 0] = STIFFNESS
    model.actuator_biasprm[motor, 1] = -STIFFNESS
    model.actuator_biasprm[motor, 2] = -DAMPING
    return model, data


def run(model, data, viewer=None, seconds=None):
    """The simulation loop. Stops when the window is closed, or after `seconds` of simulated time."""
    motor = model.actuator(JOINT).id
    joint = data.joint(JOINT)
    limit = model.jnt_actfrcrange[model.joint(JOINT).id][1]  # the most this motor may push, in newton-metres
    home_angle = data.ctrl[motor]
    next_report = 0.0
    largest_gap = 0.0
    largest_mismatch = 0.0
    hit_limit = False
    wall_start = time.time()

    print("  time   commanded   actual      gap  |   spring   damper     push")
    print("   (s)       (rad)    (rad)    (rad)  |    (N·m)    (N·m)    (N·m)")

    while (viewer is None or viewer.is_running()) and (seconds is None or data.time <= seconds + 1e-9):
        # 1. Decide the command, exactly as in 01.
        target = home_angle + SWING * math.sin(2 * math.pi * data.time / PERIOD)
        data.ctrl[motor] = target

        # 2. Work out by hand the push the motor's rule calls for at this instant.
        actual = joint.qpos[0]
        gap = target - actual
        spring = STIFFNESS * gap          # pulls toward the commanded angle, harder the bigger the gap
        damper = -DAMPING * joint.qvel[0]  # resists motion, harder the faster the joint is turning
        push = max(-limit, min(limit, spring + damper))  # the motor cannot push harder than its limit
        hit_limit = hit_limit or abs(spring + damper) > limit
        largest_gap = max(largest_gap, abs(gap))

        # 3. Print it eight times per swing.
        if data.time >= next_report - 1e-9:
            print(f"{data.time:6.3f}      {target:+.3f}   {actual:+.3f}  {gap:+.4f}  | {spring:+8.1f} {damper:+8.1f} {push:+8.1f}")
            next_report += PERIOD / 8

        # 4. Advance the physics by one timestep, then check our sum against the push
        #    the simulator itself applied during that step.
        mujoco.mj_step(model, data)
        largest_mismatch = max(largest_mismatch, abs(push - joint.qfrc_actuator[0]))

        # 5. With a window open: draw the new state, then wait for the real clock to catch up
        #    with simulated time, so the motion plays at real speed.
        if viewer is not None:
            viewer.sync()
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

    top_speed = SWING * 2 * math.pi / PERIOD
    print(f"largest gap: {largest_gap:.4f} rad ({math.degrees(largest_gap):.1f} degrees)")
    print(f"rough prediction, DAMPING / STIFFNESS x top commanded speed: {DAMPING / STIFFNESS * top_speed:.4f} rad")
    print(f"our spring + damper sum matched the simulator's own motor push to within {largest_mismatch:.6f} N·m")
    if hit_limit:
        print(f"the rule asked for more than the motor's limit of {limit:.0f} N·m at some point, so the push was capped")


def main():
    parser = argparse.ArgumentParser(description="See the spring-and-damper rule inside one FR3 motor.")
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
