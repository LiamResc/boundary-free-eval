"""The block world: a block that slides along a straight track, pushed by an ideal pusher.

Its two hidden knobs are the friction coefficient and the block's mass. Friction can push back
with up to friction x mass x g. Below that the block sticks; above it the block slides, and

    push = mass x acceleration + friction x mass x g
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_FILE = Path(__file__).with_name("block_world.xml")
GRAVITY = 9.81  # metres per second squared, the simulator's default

# Everything the next step depends on that MuJoCo keeps in the data: for the block world, 18 numbers.
SAVED_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class Snapshot:
    """Everything needed to carry on exactly from one moment of the block world."""
    state: np.ndarray  # MuJoCo's saved state: the clock, position, speed, push command and bookkeeping
    friction: float    # the hidden knobs live in the model, not the data, so MuJoCo's state leaves them out
    mass: float


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

    # ---- 4. Save, restore and branch ------------------------------------------

    def snapshot(self):
        """Save this moment: MuJoCo's state plus the hidden knobs."""
        state = np.empty(mujoco.mj_stateSize(self.model, SAVED_STATE))
        mujoco.mj_getState(self.model, self.data, state, SAVED_STATE)
        return Snapshot(state=state, friction=self._friction, mass=self.mass)

    def restore(self, snapshot):
        """Go back to a saved moment and carry on exactly from there.

        Readings worked out during a step, such as applied_push, refer to the last step taken
        and update on the next one.
        """
        self.set_friction(snapshot.friction)
        self.set_mass(snapshot.mass)
        mujoco.mj_setState(self.model, self.data, snapshot.state, SAVED_STATE)

    def branch(self):
        """A brand-new world at this same moment, for a "what if" run.

        It has its own model and data, so nothing done in the branch, not even changing its friction or
        mass, can leak back into this world.
        """
        twin = BlockWorld(friction=self._friction, mass=self.mass)
        twin.restore(self.snapshot())
        return twin

    # ---- 5. For checking and drawing only (controllers must not use this) ----

    @property
    def friction_force(self):
        """The force friction applied on the last step, in newtons; it always opposes the push or the motion.

        At the very ends of the track it also includes the end stops.
        """
        return float(self.data.qfrc_constraint[self._dof])
