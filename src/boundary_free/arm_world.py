"""The arm world: the FR3 arm with a ball-tipped pusher, and the block on its track in front of it.

The block is the block world's block. It slides along one straight track, its friction is the track joint's
friction (exactly friction x mass x g while it slides), and its two hidden knobs are the friction coefficient and its
mass. Here the track runs across the front of the robot, along y, and the arm takes the ideal pusher's place: it
moves a pusher whose ball tip shoves the block. A ball pushes on the block's flat face along the face's normal
however the hand is tilted, so the hand's orientation does not matter.

The scene is built in code from the unmodified FR3 model in models/franka_fr3.

What touches what (collision filtering, through MuJoCo's contype and conaffinity bitmasks):
  - the arm and the pusher touch the table and the block, as on the real robot;
  - the block never touches the table, so the track's friction is the only friction it feels;
  - the arm does not touch itself, and the floor is only for looking at.
"""
from pathlib import Path

import mujoco
import numpy as np

from boundary_free.block_world import GRAVITY, SAVED_STATE, Snapshot

ARM_SCENE = Path(__file__).resolve().parents[2] / "models" / "franka_fr3" / "scene.xml"

# The scene, in metres: x points away from the robot, y to its left, z up. The robot stands at the origin, bolted
# to the table, so the table top is level with the robot's base (z = 0).
TABLE_HEIGHT = 0.75
TABLE_CENTRE_X = 0.35                  # the table reaches from just behind the base to 0.85 m in front of it
TABLE_HALF_SIZE = (0.5, 0.7, TABLE_HEIGHT / 2)
TRACK_X = 0.5                          # the block's track runs across the table, this far in front of the robot
TRACK_AXIS = np.array([0.0, 1.0, 0.0])  # the direction the block slides in: along y, across the robot's front
TRACK_HALF_LENGTH = 0.45               # the block can slide this far either side of the middle
BLOCK_HALF_SIZE = 0.05                 # a 10 cm cube, as in the block world
ROD_LENGTH, ROD_RADIUS = 0.10, 0.01    # the pusher: a rod sticking out of the flange...
BALL_RADIUS = 0.02                     # ...ending in a ball whose centre is ROD_LENGTH + BALL_RADIUS from the flange
ROD_MASS, BALL_MASS = 0.10, 0.05       # kilograms
# How quickly MuJoCo pushes the pusher and the block apart when they overlap, in seconds. MuJoCo's default, 0.02, is
# springy: under a push the ball sinks 2-3 mm into the block, and the spring keeps the block in contact as the ball
# starts braking, so the block is carried instead of released. A hard ball on a block is far stiffer than that.
# (0.005 is above the simulator's stability floor of two timesteps.)
CONTACT_TIME_CONSTANT = 0.005

# Collision groups. Two shapes touch if one's contype shares a bit with the other's conaffinity.
ARM, TABLE, BLOCK = 1, 2, 4

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


def build_scene():
    """The arm world's model, built in code on top of the FR3 scene (arm, floor and light), ready to compile."""
    # 1. Start from the FR3 scene file, unmodified on disk.
    spec = mujoco.MjSpec.from_file(str(ARM_SCENE))

    # 2. Gravity compensation: a computed push holds up each part of the arm's weight, and it comes out of the
    #    joint's motor, as on the real robot. Done before the block is added, so the block gets none.
    for body in spec.bodies:
        body.gravcomp = 1.0
    for joint in spec.joints:
        joint.actgravcomp = True

    # 3. The pusher, mounted where a tool goes (attachment_site, on the flange, its z axis pointing out of the
    #    flange): a rod ending in a ball, with a site at the ball's centre for steering it.
    flange = spec.site("attachment_site")
    pusher = spec.body("fr3_link7").add_body(name="pusher", pos=flange.pos, quat=flange.quat, gravcomp=1.0)
    pusher.add_geom(name="pusher_rod", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, ROD_LENGTH],
                    size=[ROD_RADIUS, 0, 0], mass=ROD_MASS, rgba=[0.3, 0.3, 0.3, 1])
    pusher.add_geom(name="pusher_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, ROD_LENGTH + BALL_RADIUS],
                    size=[BALL_RADIUS, 0, 0], mass=BALL_MASS, rgba=[0.2, 0.2, 0.2, 1])
    pusher.add_site(name="pusher_tip", pos=[0, 0, ROD_LENGTH + BALL_RADIUS])

    # 4. The table, under the robot and the workspace; the floor moves down below it and is only for looking at.
    floor = spec.geom("floor")
    floor.pos = [0, 0, -TABLE_HEIGHT]
    floor.contype = floor.conaffinity = 0
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=TABLE_HALF_SIZE,
                            pos=[TABLE_CENTRE_X, 0, -TABLE_HEIGHT / 2], rgba=[0.55, 0.45, 0.35, 1])

    # 5. The block, resting on the table on its track: a slide joint along TRACK_AXIS. As in the block world, the
    #    two "friction" settings make the joint's friction grip firmly when the block is meant to stick.
    block = spec.worldbody.add_body(name="block", pos=[TRACK_X, 0, BLOCK_HALF_SIZE])
    block.add_joint(name="track", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=TRACK_AXIS,
                    range=[-TRACK_HALF_LENGTH, TRACK_HALF_LENGTH], solref_friction=[0.004, 1],
                    solimp_friction=[0.99, 0.999, 0.001, 0.5, 2])
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF_SIZE] * 3, mass=1.0,
                   rgba=[0.85, 0.35, 0.2, 1])

    # 6. What touches what. The FR3's collision shapes (group 3) and the pusher are ARM; they touch whatever lists
    #    ARM in its conaffinity, which the table and the block both do. The block's own bits never meet the table's,
    #    and the arm's conaffinity is empty, so the arm never touches itself.
    for geom in spec.geoms:
        if geom.group == 3 or geom.name.startswith("pusher"):
            geom.contype, geom.conaffinity = ARM, 0
    table, block_geom = spec.geom("table"), spec.geom("block")
    table.contype, table.conaffinity = TABLE, ARM
    block_geom.contype, block_geom.conaffinity = BLOCK, ARM

    # 7. A stiff contact between the pusher and the block, so a shove releases the block cleanly.
    for name in ("pusher_rod", "pusher_ball", "block"):
        spec.geom(name).solref = [CONTACT_TIME_CONSTANT, 1.0]
    return spec


class ArmWorld:
    """The FR3 with a ball-tipped pusher, and the block on its track with its two hidden knobs: friction and mass."""

    def __init__(self, friction=0.3, mass=1.0):
        self.model = build_scene().compile()
        self.data = mujoco.MjData(self.model)
        self.block_body = self.model.body("block").id
        self._track_qpos = self.model.joint("track").qposadr[0]
        self._track_dof = self.model.joint("track").dofadr[0]
        self._arm_qpos = np.array([self.model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.arm_dofs = np.array([self.model.joint(name).dofadr[0] for name in ARM_JOINTS])  # their Jacobian columns
        self._tip = self.model.site("pusher_tip").id
        self._block_geom = self.model.geom("block").id
        self._pusher_geoms = {self.model.geom("pusher_rod").id, self.model.geom("pusher_ball").id}

        # Start at the home pose, with the motors commanded to hold it.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.home = self.data.qpos[self._arm_qpos].copy()
        self.command(self.home)

        self._friction = friction
        self.set_mass(mass)  # also sets the friction to match the block's weight
        mujoco.mj_forward(self.model, self.data)

    # ---- 1. The block's hidden knobs (as in the block world) ------------------

    @property
    def friction(self):
        """The friction coefficient between block and table (no units)."""
        return self._friction

    @property
    def mass(self):
        """The block's mass, in kilograms."""
        return float(self.model.body_mass[self.block_body])

    @property
    def friction_limit(self):
        """The most friction can push back with, friction x mass x g, in newtons."""
        return self._friction * self.mass * GRAVITY

    def set_friction(self, friction):
        """Change the friction coefficient, at any time."""
        self._friction = friction
        self._apply_friction_limit()

    def set_mass(self, mass):
        """Change the block's mass, at any time."""
        scale = mass / self.mass
        self.model.body_mass[self.block_body] = mass
        self.model.body_inertia[self.block_body] *= scale
        # Recompute the numbers the simulator works out from the masses, on a throwaway copy of the data:
        # mj_setConst moves the data it is given back to the start position.
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self._apply_friction_limit()

    def _apply_friction_limit(self):
        # The track's friction holds the block still up to this force, and pushes back with exactly this force
        # while the block slides.
        self.model.dof_frictionloss[self._track_dof] = self.friction_limit

    # ---- 2. Commanding the arm and stepping -----------------------------------

    def command(self, joint_angles):
        """Command the seven motors to these joint angles, in radians: each motor's PD control law then pushes its
        joint towards its angle, on top of the gravity compensation."""
        self.data.ctrl[:] = joint_angles

    def step(self):
        """Advance the simulation by one timestep, then work out where every part of the arm now is, so that readings
        such as tip_position, and Jacobians, describe the arm after the step rather than before it."""
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    # ---- 3. What can be observed ----------------------------------------------

    @property
    def time(self):
        """Simulated time, in seconds."""
        return float(self.data.time)

    @property
    def timestep(self):
        """How much simulated time one step covers, in seconds."""
        return float(self.model.opt.timestep)

    @property
    def position(self):
        """Where the block is along its track, in metres from the middle (positive is the robot's left)."""
        return float(self.data.qpos[self._track_qpos])

    @property
    def velocity(self):
        """The block's speed along its track, in metres per second."""
        return float(self.data.qvel[self._track_dof])

    @property
    def joint_angles(self):
        """The arm's seven joint angles, in radians, as its encoders would read them."""
        return self.data.qpos[self._arm_qpos].copy()

    @property
    def joint_velocities(self):
        """The arm's seven joint speeds, in radians per second."""
        return self.data.qvel[self.arm_dofs].copy()

    @property
    def tip_position(self):
        """Where the centre of the pusher's ball is, in metres (x, y, z), worked out from the joint angles."""
        return self.data.site_xpos[self._tip].copy()

    # ---- 4. Save, restore and branch (as in the block world) ------------------

    def snapshot(self):
        """Save this moment: MuJoCo's state (the arm's joints and the block, their speeds, the motor commands and
        bookkeeping) plus the block's hidden knobs, which live in the model, not the data."""
        state = np.empty(mujoco.mj_stateSize(self.model, SAVED_STATE))
        mujoco.mj_getState(self.model, self.data, state, SAVED_STATE)
        return Snapshot(state=state, friction=self._friction, mass=self.mass)

    def restore(self, snapshot):
        """Go back to a saved moment and carry on exactly from there."""
        self.set_friction(snapshot.friction)
        if snapshot.mass != self.mass:
            self.set_mass(snapshot.mass)
        mujoco.mj_setState(self.model, self.data, snapshot.state, SAVED_STATE)
        mujoco.mj_kinematics(self.model, self.data)  # so tip_position and Jacobians describe the restored arm
        mujoco.mj_comPos(self.model, self.data)

    def branch(self):
        """A brand-new arm world at this same moment, for a "what if" run. It has its own model and data, so nothing
        done in the branch can leak back into this world."""
        twin = ArmWorld(friction=self._friction, mass=self.mass)
        twin.restore(self.snapshot())
        return twin

    # ---- 5. For checking and drawing only (controllers must not use this) ----

    @property
    def friction_force(self):
        """The force the track's friction applied to the block on the last step, in newtons, along the track."""
        rows = ((self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
                & (self.data.efc_id == self._track_dof))
        return float(self.data.efc_force[rows].sum())

    @property
    def pusher_touches_block(self):
        """Whether the pusher (its ball or its rod) was touching the block on the last step."""
        return any({c.geom1, c.geom2} & self._pusher_geoms and self._block_geom in (c.geom1, c.geom2)
                   for c in self.data.contact[:self.data.ncon])
