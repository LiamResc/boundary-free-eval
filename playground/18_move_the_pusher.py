"""Move the pusher: get the ball ready to shove the block from one side, then cross over to the other side and back.

With a window, where a red dot marks the reference (on macOS this needs the `mjpython` launcher):
    uv run mjpython playground/18_move_the_pusher.py
Numbers only:
    uv run python playground/18_move_the_pusher.py --headless

For each move it prints how closely the ball tracked the reference, how close to the ready point it ended, the
fastest any joint turned as a share of the real FR3's limit, and whether the block was touched.
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from boundary_free.arm_world import BALL_RADIUS, BLOCK_HALF_SIZE, ArmWorld
from boundary_free.motion import JOINT_SPEED_LIMITS, PusherMotion, Trajectory, ready_point, trajectory_to_ready
from boundary_free.viewer_tools import RealTime

# ---- Things to change and re-run ---------------------------------------------
SPEED = 0.3          # how fast the ball moves, m/s
FEEDFORWARD = True   # False removes the velocity feedforward: the ball then trails the reference by 0.1 s x speed
MOVES = (+1, -1, +1)  # each move gets ready to push the block this way: +1 towards +y (the robot's left), -1 back
# -------------------------------------------------------------------------------

SETTLE = 0.5  # seconds to hold still after each move


def run(world, viewer=None):
    motion = PusherMotion(world)
    if not FEEDFORWARD:
        motion.lead[:] = 0.0
    pace = RealTime(world)
    print(f"ball speed {SPEED} m/s, velocity feedforward {'on' if FEEDFORWARD else 'off'}\n")
    print("  move              time   tracking error (mm)   end error   fastest joint      lowest over    block")
    print("                     (s)    mean while moving      (mm)     (share of limit)   the block (mm)  touched")
    for direction in MOVES:
        corners = trajectory_to_ready(world.tip_position, world.position, direction)
        # The same corners, covered at SPEED.
        trajectory = Trajectory.through([s.start for s in corners.segments] + [corners.end], top_speed=SPEED)
        motion.follow(trajectory)
        errors, speeds, over, touched = [], [], [], False
        end = world.time + trajectory.duration + SETTLE
        while world.time < end - 1e-9:
            error = motion.update()
            world.step()
            if not motion.done:
                errors.append(np.linalg.norm(error))
            speeds.append(np.abs(world.joint_velocities) / JOINT_SPEED_LIMITS)
            tip = world.tip_position
            if abs(tip[1] - world.position) < BLOCK_HALF_SIZE + BALL_RADIUS:
                over.append(tip[2] - BALL_RADIUS - 2 * BLOCK_HALF_SIZE)  # the ball's bottom above the block's top
            touched |= any("block" in (world.model.geom(c.geom1).name, world.model.geom(c.geom2).name)
                           for c in world.data.contact[:world.data.ncon])
            if viewer is not None:
                draw_reference(viewer, motion.trajectory.reference(world.time - motion._started)[0])
                viewer.sync()
                pace()
        end_error = np.linalg.norm(world.tip_position - ready_point(world.position, direction))
        lowest = f"{1000 * min(over):6.1f}" if over else "     -"
        print(f"  ready to push {'+y' if direction > 0 else '-y'}  {trajectory.duration:5.2f}   "
              f"{1000 * np.mean(errors):8.2f}              {1000 * end_error:6.2f}       "
              f"{100 * np.max(speeds):5.1f}%          {lowest}       {'yes' if touched else 'no'}")


def draw_reference(viewer, point):
    """Put a small red dot at the reference in the viewer."""
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0],
                        pos=point, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.8])
    viewer.user_scn.ngeom = 1


def main():
    parser = argparse.ArgumentParser(description="Move the FR3's pusher around the block.")
    parser.add_argument("--headless", action="store_true", help="no window: run the moves once and print")
    args = parser.parse_args()
    world = ArmWorld()
    if args.headless:
        run(world)
    else:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            run(world, viewer=viewer)


if __name__ == "__main__":
    main()
