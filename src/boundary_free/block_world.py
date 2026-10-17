"""The block world: a block that slides along a straight track, pushed by an ideal pusher.

Its two hidden knobs are the friction coefficient and the block's mass. Friction can push back
with up to friction x mass x g. Below that the block sticks; above it the block slides, and

    push = mass x acceleration + friction x mass x g
"""
from pathlib import Path

import mujoco

MODEL_FILE = Path(__file__).with_name("block_world.xml")
GRAVITY = 9.81  # metres per second squared, the simulator's default


class BlockWorld:
    """A block on a straight track, with an ideal pusher and two hidden knobs: friction and mass."""

    def __init__(self, friction=0.3, mass=1.0):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
        self.data = mujoco.MjData(self.model)
        self._body = self.model.body("block").id
        self._dof = self.model.joint("track").dofadr[0]
        self._friction = friction
        self.set_mass(mass)  # also sets the friction to match the block's weight
        mujoco.mj_forward(self.model, self.data)

    # ---- 1. The hidden knobs --------------------------------------------------

    @property
    def friction(self):
        """The friction coefficient between block and table (no units)."""
        return self._friction

    @property
    def mass(self):
        """The block's mass, in kilograms."""
        return float(self.model.body_mass[self._body])

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
        self.model.body_mass[self._body] = mass
        self.model.body_inertia[self._body] *= scale  # a heavier block of the same size is harder to turn, too

        # The simulator keeps some numbers worked out from the masses, so recompute them.
        # mj_setConst uses the data it is given as scratch space and moves it back to the start position,
        # so it gets a throwaway copy, never the running simulation.
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))

        self._apply_friction_limit()  # a heavier block presses harder on the table, so more friction

    def _apply_friction_limit(self):
        # The joint's friction holds the block still up to this force, and pushes back with exactly
        # this force while the block slides.
        self.model.dof_frictionloss[self._dof] = self.friction_limit

    # ---- 2. Pushing and stepping ----------------------------------------------

    @property
    def push_limit(self):
        """The strongest push (or pull) the pusher can give, in newtons."""
        return float(self.model.actuator_ctrlrange[0, 1])

    def push(self, force):
        """Set the pusher's force in newtons: positive pushes along +x, negative pulls. Capped at push_limit."""
        self.data.ctrl[0] = force

    def step(self):
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

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
        """Where the block is along the track, in metres from the middle of the table."""
        return float(self.data.qpos[0])

    @property
    def velocity(self):
        """The block's speed along the track, in metres per second (negative means moving towards -x)."""
        return float(self.data.qvel[0])

    @property
    def applied_push(self):
        """The force the pusher actually applied on the last step, after capping, in newtons."""
        return float(self.data.actuator_force[0])
