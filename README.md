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