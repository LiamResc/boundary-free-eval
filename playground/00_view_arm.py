"""Open the FR3 arm in MuJoCo's interactive viewer, starting from its "home" pose.

Run from the project root:
    uv run python playground/00_view_arm.py
"""
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_fr3" / "scene.xml"

# 1. The MODEL is the description of the world: which bodies exist, how they are
#    jointed together, their masses, the motors. It stays fixed while we simulate.
model = mujoco.MjModel.from_xml_path(str(SCENE))

# 2. The DATA is the situation right now: every joint angle and speed, the commands
#    going to the motors, the clock. Simulating means updating the data over and over.
data = mujoco.MjData(model)

# 3. Fresh data has every joint at angle zero, which two of the FR3's joints are not
#    allowed to reach. The model file stores a sensible "home" pose, so start from that.
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

# 4. Hand both to the viewer. It runs the simulation itself until the window is closed.
mujoco.viewer.launch(model, data)
