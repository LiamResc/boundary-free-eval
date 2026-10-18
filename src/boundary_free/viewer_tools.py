"""Helpers for watching the block world in MuJoCo's viewer window."""
import time

import mujoco
import numpy as np

BLOCK_HALF = 0.05    # the block is a 10 cm cube
BLOCK_HEIGHT = 0.8   # height of the block's centre above the floor, in metres
TABLE_TOP = 0.75     # height of the table top, in metres
ARROW_SCALE = 0.04   # metres of arrow per newton of force


def draw_forces(viewer, world, target=None):
    """Draw the forces on the block as arrows: red for the pusher, blue for friction (4 cm per newton).

    Neither force comes from anything visible: the pusher is an invisible motor on the block's track,
    and friction is set on the track too. If a target is given, a green marker shows it.
    """
    scene = viewer.user_scn
    scene.ngeom = 0
    x = world.position

    # Red: the pusher's force, drawn as a push arriving at the back face of the block.
    push = world.applied_push
    if abs(push) > 0.01:
        direction = np.sign(push)
        tip = np.array([x - direction * BLOCK_HALF, 0.0, BLOCK_HEIGHT])
        tail = tip - np.array([direction * ARROW_SCALE * abs(push), 0.0, 0.0])
        _add_arrow(scene, tail, tip, [0.9, 0.1, 0.1, 1.0])

    # Blue: friction, drawn beside the block at the table top, where the rubbing happens.
    friction = world.friction_force
    if abs(friction) > 0.01:
        tail = np.array([x, -(BLOCK_HALF + 0.05), TABLE_TOP + 0.01])
        tip = tail + np.array([ARROW_SCALE * friction, 0.0, 0.0])
        _add_arrow(scene, tail, tip, [0.1, 0.4, 0.9, 1.0])

    # Green: the target, as a thin post standing on the table.
    if target is not None:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.004, 0.06, 0.0]),
                            np.array([target, 0.0, TABLE_TOP + 0.06]), np.eye(3).flatten(),
                            np.array([0.1, 0.8, 0.2, 1.0], dtype=np.float32))
        scene.ngeom += 1


def _add_arrow(scene, tail, tip, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, tail, tip)
    scene.ngeom += 1


class RealTime:
    """Keeps a window's playback at real speed: call it once per simulation step."""

    def __init__(self, world):
        self.world = world
        self.sim_start = world.time
        self.wall_start = time.time()

    def __call__(self):
        ahead = (self.world.time - self.sim_start) - (time.time() - self.wall_start)
        if ahead > 0:
            time.sleep(ahead)
