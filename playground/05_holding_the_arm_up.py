"""Two ways to hold the arm up against its own weight: a gap in the spring, or a computed push.

    uv run python playground/05_holding_the_arm_up.py
"""
from pathlib import Path

import mujoco

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"
HAND = "attachment_site"  # the point at the end of the arm's last link, where a tool gets mounted

# ---- Things to change and re-run ---------------------------------------------
COMPENSATE = 0.8  # share of the arm's weight cancelled by a computed push: 0.0 is none of it, 1.0 is all of it
# -------------------------------------------------------------------------------


def load():
    """Build the model with gravity compensation set to COMPENSATE, and the data at the home pose."""
    # Reading the scene as a "spec" lets us change it before it becomes a model, without editing the file.
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for body in spec.bodies:
        body.gravcomp = COMPENSATE  # the simulator will push each part of the arm up with this share of its weight
    model = spec.compile()

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def tidy(value, places):
    """A signed number with a fixed number of decimals (and no '-0.0')."""
    return f"{round(value, places) + 0.0:+.{places}f}"


def main():
    model, data = load()
    hand_before = data.site(HAND).xpos.copy()

    # Ask every motor to hold its home angle, and let three seconds pass so the arm settles.
    while data.time < 3.0:
        mujoco.mj_step(model, data)

    print(f"Share of the arm's weight cancelled by a computed push: {100 * COMPENSATE:.0f}%")
    print()
    print("              push needed     computed     spring")
    print("              to hold it      push         push         gap")
    print("              (N·m)           (N·m)        (N·m)        (rad)")
    for j in range(model.njnt):
        needed = data.qfrc_bias[j]        # what it takes to hold this joint still against the arm's weight
        computed = data.qfrc_gravcomp[j]  # the gravity-compensation push, worked out from the arm's known weight
        spring = data.qfrc_actuator[j]    # the push from the motor's spring-and-damper rule
        gap = data.ctrl[j] - data.qpos[j]
        print(f"{model.joint(j).name}    {tidy(needed, 1):>7}        {tidy(computed, 1):>7}      {tidy(spring, 1):>7}      {tidy(gap, 4)}")

    sunk = 1000 * (hand_before[2] - data.site(HAND).xpos[2])
    print()
    print(f"The hand has sunk {sunk:.1f} mm.")


if __name__ == "__main__":
    main()
