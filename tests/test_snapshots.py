"""Checks that saving, restoring and branching the block world are exact."""
from boundary_free.block_world import BlockWorld
from boundary_free.controllers import OracleController, static_controller
from boundary_free.simulate import drive

TARGET = 0.3


def sliding_world():
    """A world partway through a push, with the block sliding."""
    world = BlockWorld(friction=0.6, mass=1.0)
    world.data.qpos[0] = -0.3
    drive(world, static_controller(), TARGET, 0.6)
    return world


def test_replaying_from_a_snapshot_is_identical():
    world = sliding_world()
    saved = world.snapshot()
    first = drive(world, static_controller(), TARGET, 2.0)
    world.restore(saved)
    second = drive(world, static_controller(), TARGET, 2.0)
    assert first.position == second.position  # equal to the last digit, not just close


def test_a_snapshot_brings_back_the_hidden_knobs():
    world = sliding_world()
    saved = world.snapshot()
    expected = drive(world.branch(), static_controller(), TARGET, 1.0)

    world.set_friction(0.2)
    world.set_mass(2.0)
    drive(world, static_controller(), TARGET, 0.5)
    world.restore(saved)

    assert (world.friction, world.mass) == (0.6, 1.0)
    assert drive(world, static_controller(), TARGET, 1.0).position == expected.position


def test_a_snapshot_does_not_change_when_the_world_moves_on():
    world = sliding_world()
    saved = world.snapshot()
    copy_of_state = saved.state.copy()
    drive(world, static_controller(), TARGET, 1.0)
    assert (saved.state == copy_of_state).all()


def test_a_branch_starts_at_the_same_moment():
    world = sliding_world()
    twin = world.branch()
    assert (twin.time, twin.position, twin.velocity) == (world.time, world.position, world.velocity)
    assert (twin.friction, twin.mass) == (world.friction, world.mass)


def test_running_a_branch_leaves_the_main_world_untouched():
    world = sliding_world()
    before = (world.time, world.position, world.velocity, world.friction, world.mass)

    twin = world.branch()
    twin.set_friction(0.2)  # even changing the branch's hidden knobs must not leak back
    drive(twin, OracleController(twin), TARGET, 1.0)

    assert (world.time, world.position, world.velocity, world.friction, world.mass) == before


def test_a_run_with_branches_matches_a_run_without_them():
    plain = BlockWorld(friction=0.6, mass=1.0)
    plain.data.qpos[0] = -0.3
    expected = drive(plain, static_controller(), TARGET, 4.0).position

    probed = BlockWorld(friction=0.6, mass=1.0)
    probed.data.qpos[0] = -0.3
    controller = static_controller()
    positions = []
    for _ in range(8):  # every half second, run a 1-second "what if" with the oracle on a branch
        twin = probed.branch()
        drive(twin, OracleController(twin), TARGET, 1.0)
        positions += drive(probed, controller, TARGET, 0.5).position

    assert positions == expected
