# boundary-free-eval

# Simulated Arm - FR3

## Motors / Joints
Each motor follows the rule:

    push = STIFFNESS * (commanded_angle - actual_angle) - DAMPING (joint_speed)

We have no simulated max speed nor max acceleration.
The **STIFFNESS** has to do with the spring. This spring pushed the join towards the angle we requested. Defaulted to **4500 newton-meters**.

Clearly, the damping force is larger when joints are moving faster. 
**DAMPING** is **450 newton meters**, by default.

Motors / joints also have friction. Each joint gets **1.1Nm** of drag (constant)
### Push
Push is a twisting force (torque).

### Calculating joint displacement to reach a target
There are 7 joints on this robot arm. The coordiante of any point is in terms of (x,y,z). There can be many answers to how a the joints should move in order for the hand to reach a certain position, and so the rule is that you pick the joint mix that involved the least amount of total turning.

