"""How does the hand move when joints turn a little? The table that answers it, and a test of the table.

    uv run python playground/04_how_the_hand_moves.py
"""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"
HAND = "attachment_site"  # the point at the end of the arm's last link, where a tool gets mounted

# ---- Things to change and re-run ---------------------------------------------
TURNS = {  # turns to make from the home pose, in radians, for as many joints as you like
    "fr3_joint2": -0.158,
    "fr3_joint4": 1.93,
}
# -------------------------------------------------------------------------------


def load():
    """Build the model (what exists) and the data (the situation right now), at the home pose."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)  # work out where every part of the arm is
    return model, data


def hand_position(data):
    """Where the hand is: x (forward), y (to the robot's left), z (up), in metres from the centre of the base."""
    return data.site(HAND).xpos.copy()


def hand_table(model, data):
    """One row per joint: how far the hand moves in x, y, z, in metres per radian, for a small turn of that joint.

    The table is worked out for the pose the arm is in right now. Its proper name is the Jacobian.
    """
    # MuJoCo fills in two tables with one column per joint. The first is about the hand's position,
    # which is what we want. The second is about which way the hand faces; we leave it for later.
    position_part = np.zeros((3, model.nv))
    facing_part = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, position_part, facing_part, model.site(HAND).id)
    return position_part.T  # flipped so that each row is one joint


def row(xyz, places=3):
    """Three numbers lined up under the x, y, z headings (and no '-0.000')."""
    gap = " " * (10 - places)
    return gap.join(f"{round(v, places) + 0.0:+.{places}f}" for v in xyz)


def main():
    model, data = load()
    joint_names = [model.joint(j).name for j in range(model.njnt)]

    # 1. The table at the home pose.
    table = hand_table(model, data)
    print("At the home pose, a small turn of each joint moves the hand by this much per radian of turn:")
    print()
    print("               x (forward)  y (left)     z (up)     metres per radian")
    for name, per_radian in zip(joint_names, table):
        print(f"{name}     {row(per_radian)}")

    # 2. What the table predicts for the turns chosen at the top: each joint's row times its turn, added up.
    predicted = sum(table[joint_names.index(name)] * turn for name, turn in TURNS.items())

    # 3. What really happens, by geometry: place the joints at their new angles and look at the hand.
    at_home = hand_position(data)
    for name, turn in TURNS.items():
        data.joint(name).qpos[0] += turn
    mujoco.mj_forward(model, data)
    actual = hand_position(data) - at_home

    # 4. Compare the two.
    print()
    print("Turning " + ", ".join(f"{name} by {turn:+.5f} rad" for name, turn in TURNS.items()) + ":")
    print()
    print("                          x (forward)  y (left)     z (up)     in metres")
    print(f"the table predicts        {row(predicted, 4)}")
    print(f"geometry says             {row(actual, 4)}")
    print(f"they differ by {1000 * np.linalg.norm(predicted - actual):.1f} mm")


if __name__ == "__main__":
    main()
