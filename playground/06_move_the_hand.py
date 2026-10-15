"""Move the hand to a chosen point in a straight line: a first demo of the "move the hand here" layer.

With a window, where a red dot marks the target point (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/06_move_the_hand.py
Numbers only:
    uv run python playground/06_move_the_hand.py --headless
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"
HAND = "attachment_site"  # the point at the end of the arm's last link, where a tool gets mounted

# ---- Things to change and re-run ---------------------------------------------
MOVE = (0.15, 0.20, -0.25)  # where to send the hand, relative to where it starts: x forward, y left, z up, in metres
SPEED = 0.1                 # how fast the target point slides along the line, in metres per second
# -------------------------------------------------------------------------------

CAUTION = 1e-4  # keeps step 4's answer sensible in awkward poses, such as the arm nearly straight
MAX_TURN = 0.2  # safety limit: the biggest turn, and so the biggest gap, any motor is handed, in radians


def load():
    """Build the model with gravity compensation on, and the data at the home pose."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for body in spec.bodies:
        body.gravcomp = 1.0        # a computed push holds up each part of the arm's weight...
    for joint in spec.joints:
        joint.actgravcomp = True   # ...and it comes out of the joint's motor, as on the real robot
    model = spec.compile()

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def layer_step(model, data, target):
    """Steps 2 to 5 of the layer: command the motors so the hand heads for the target point."""
    # 2. The miss: the target point minus where the hand is now. Three numbers, in metres.
    miss = target - data.site(HAND).xpos

    # 3. A fresh table for the arm's current pose: how the hand moves per radian of each joint.
    #    MuJoCo fills it in with one column per joint (and a second table, about facing, that we skip for now).
    table = np.zeros((3, model.nv))
    facing_table = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, table, facing_table, model.site(HAND).id)

    # 4. The least-turning joint turns that the table says would cover the miss.
    #    (The pseudo-inverse formula. CAUTION adds a tiny amount to keep the answer calm in awkward poses.)
    turns = table.T @ np.linalg.solve(table @ table.T + CAUTION * np.eye(3), miss)
    biggest = np.abs(turns).max()
    if biggest > MAX_TURN:
        turns *= MAX_TURN / biggest  # scale every turn down together, so the direction stays the same

    # 5. Command each motor to its joint's current angle plus its turn.
    data.ctrl[:] = data.qpos + turns
    return miss


def draw_target(viewer, target):
    """Put a small red dot at the target point in the viewer."""
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0],
                        pos=target, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.8])
    viewer.user_scn.ngeom = 1


def run(model, data, viewer=None):
    """Slide the target point from the hand's start to the goal, and let the layer chase it."""
    start = data.site(HAND).xpos.copy()
    goal = start + np.array(MOVE)
    distance = np.linalg.norm(goal - start)
    direction = (goal - start) / distance
    move_time = distance / SPEED
    stop_time = move_time + 1.5  # carry on briefly after the target point reaches the goal
    next_report = 0.0
    misses_while_moving = []
    wall_start = time.time()

    print(f"moving the hand {100 * distance:.1f} cm at {100 * SPEED:.0f} cm/s ({move_time:.1f} s)")
    print()
    print("  time   how far along   miss (mm)")
    while (viewer is None and data.time <= stop_time) or (viewer is not None and viewer.is_running()):
        # 1. Slide the target point along the line, at SPEED, stopping at the goal.
        target = start + direction * min(SPEED * data.time, distance)

        # 2 to 5. The layer works out the commands, then the physics moves the arm one step.
        miss = layer_step(model, data, target)
        mujoco.mj_step(model, data)

        if 0.5 < data.time < move_time:
            misses_while_moving.append(np.linalg.norm(miss))
        if data.time >= next_report - 1e-9 and data.time <= stop_time:
            along = np.dot(data.site(HAND).xpos - start, direction) / distance
            print(f"{data.time:6.2f}       {100 * along:5.1f}%       {1000 * np.linalg.norm(miss):6.2f}")
            next_report += 0.25

        if viewer is not None:
            draw_target(viewer, target)
            viewer.sync()
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

    print()
    if misses_while_moving:
        print(f"while moving, the hand missed the target point by {1000 * np.mean(misses_while_moving):.1f} mm on average")
    print(f"at the end, the hand is {1000 * np.linalg.norm(goal - data.site(HAND).xpos):.2f} mm from the goal")


def main():
    parser = argparse.ArgumentParser(description="Move the FR3's hand to a point in a straight line.")
    parser.add_argument("--headless", action="store_true", help="no window: run the move once and print")
    args = parser.parse_args()

    model, data = load()
    if args.headless:
        run(model, data)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(model, data, viewer=viewer)


if __name__ == "__main__":
    main()
