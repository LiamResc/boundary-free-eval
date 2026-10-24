# boundary-free-eval
A robot cannot be made general purpose by training it once on a large collection of data and shipping it into the real world, because the real world is too big and too changeable for any fixed dataset to cover, so a robot has to keep improving from its own experience after deployment, and the field needs a new kind of test to tell whether it does.

This idea is in contrast to traditional LLM training and deployment. LLM's can get close to general purpose through massive sums of data for extensive pretraining. 

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


## Saving state

To support the later regret based oracle vs real comparison, we must be able to save the state of the simulation (at any given step t) exactly. That way, we can later restore and plug in the oracle controller. 

What must be saved?

### MuJoCo's state
This includes the clock, the blocks position and speed, and more (forces on bodies, last push command, etc)
MuJoCo's state does NOT cover the model, it only covers the data [1]

### Model data
This includes friction and mass.

### Later additions
Anything with memory, such as the streams schedule.



### Model vs Data in MuJoCo
The *model* is what the world is: What exists, how the world is built, the worlds rules.
It typically does NOT change with time (static). 
However, for our experiment, we will be editing the model mid run!
Described in XML

The *data* is what is happening in the world at a given time step. 

## The Stream
The stream is a single "long" continuous run with no resets. Push targets will arrive one after another, and friction and mass change along the way on a hidden schedule.
All the controller can see is the blocks position, its speed and the current target.

# Regret

Regret is measured at a moment in time, and it tells you how much better the oracle controller would have done if it had taken over right then. 

General definition (add citation):

\begin{aligned}
R(T) &= \max_{d \in \mathcal{D}} \sum_{t=1}^{T} \Big[\, J_H\big(d(\pi_t) \mid h_t\big) - J_H\big(\pi_t \mid h_t\big) \,\Big] \\[6pt]
J_H(\pi \mid h_t) &= \mathbb{E}_{\pi}\!\left[\, \sum_{k=0}^{H-1} \gamma^{k} R_{t+k+1} \;\middle|\; h_t \right]
\end{aligned}


For our case, we consider only the discounted sum up to 4s into the future. 

## Deviation
A deviation is one specific "what if you acted differently here" alternative that you test your agent against. 
It is a rule, **d**, that takes in the agents policy at time t, **π_t**, and gives back a different policy, **d(π_t)**, used from t onwards.

You measure regret against a deviation (how much better d(π_t) would have done compared to what π_t did)
