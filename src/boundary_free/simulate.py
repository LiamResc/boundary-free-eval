"""Running a controller on the block world and recording what happens."""
from dataclasses import dataclass, field


@dataclass
class Record:
    """What happened on each step: lists of equal length."""
    time: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    push: list = field(default_factory=list)


def drive(world, controller, target, seconds, on_step=None):
    """Let the controller push the block towards a target for a while, recording each step.

    on_step, if given, is called after every step (for example, to update a window).
    """
    record = Record()
    end = world.time + seconds
    while world.time < end - 1e-9:
        # 1. The controller sees what a real robot could see, and decides the push.
        world.push(controller.push(world.position, world.velocity, target))

        # 2. The physics moves the block one step.
        world.step()

        # 3. Write down what happened.
        record.time.append(world.time)
        record.position.append(world.position)
        record.velocity.append(world.velocity)
        record.push.append(world.applied_push)
        if on_step is not None:
            on_step()
    return record
