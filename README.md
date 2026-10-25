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

$$
R(T) = \max_{d \in \mathcal{D}} \sum_{t=1}^{T} \Big[ J_H\big(d(\pi_t) \mid h_t\big) - J_H\big(\pi_t \mid h_t\big) \Big]
$$

where

$$
J_H(\pi \mid h_t) = \mathbb{E}_{\pi}\left[ \sum_{k=0}^{H-1} \gamma^{k} R_{t+k+1} \enspace\middle|\enspace h_t \right]
$$

and:

- $t$ is the **time step**, and $T$ is the number of time steps so far.
- $\pi_t$ is the agent's **policy** at time step $t$, its rule for choosing actions. It has a subscript because a learning agent's policy changes over time.
- $h_t$ is the **history** up to time step $t$: every observation, action and reward so far. Both returns in the bracket are conditioned on the same $h_t$, so they start from exactly the same situation, and mistakes made before $t$ cancel out.
- $d$ is a **deviation**, a rule that turns the agent's policy into an alternative policy $d(\pi_t)$ (see Deviation below).
- $\mathcal{D}$ is the **set of deviations** the agent is evaluated against. The max sits outside the sum, so it picks the single deviation with the largest total regret over the whole stream.
- $J_H(\pi \mid h_t)$ is the **expected discounted return** of a policy $\pi$, either $\pi_t$ or $d(\pi_t)$, over the next $H$ steps, starting from $h_t$. It plays the role of the value function $v_\pi(s)$ in Sutton & Barto, but starts from a history instead of a state and stops after $H$ steps.
- $H$ is the **horizon**, the number of steps the return looks ahead, and $k$ counts those steps from $0$ to $H-1$.
- $R_{t+k+1}$ is the **reward** received $k+1$ steps after $t$. It is unrelated to $R(T)$, which is regret.
- $\gamma$ is the **discount factor** ($0 \le \gamma \lt 1$). A reward $k$ steps ahead is weighted by $\gamma^k$, so nearer rewards count more.
- $\mathbb{E}_\pi$ is the **expectation** over the futures that can unfold when following $\pi$, in case the environment or the policy is random.
- $R(T)$ is the **cumulative regret** up to time step $T$, and $R(T)/T$ is the **average regret** (the regret rate).

In this project:

- The agent's policy $\pi_t$ is the static controller, whose policy never changes. The adaptive controller, whose policy changes as its estimates do, comes next.
- $\mathcal{D}$ holds one deviation, "hand control to the oracle from here", so the max has nothing to choose.
- The reward is the negative of the miss (the block's distance from its target), in metres: $R_{t+1} = -\lvert \text{target} - \text{position} \rvert$.
- One time step is 0.002 s, and we consider only the discounted sum up to 4 s into the future: $H = 2000$ steps (`LOOK_AHEAD` in `regret.py`).
- The discount factor is $\gamma = e^{-0.002} \approx 0.998$ per step (`FADE_TIME` = 1 s), so a reward 1 s ahead counts about 0.37 as much as the first one.
- $h_t$ is represented by a snapshot of the simulator. The deviation's return comes from restoring the snapshot on a branch and letting the oracle run (a rollout). Elelimy et al. estimate it with importance sampling instead, because the real world cannot be rewound.
- The expectation drops out: the simulator is deterministic, so one rollout gives the expected return exactly.
- Each return is divided by $\sum_{k=0}^{H-1} \gamma^k$, so it reads as a weighted average miss in metres. This rescales regret but never changes its sign.
- The sum over $t$ runs over checkpoints every 0.5 s (`CHECK_EVERY`), 233 of them in the 120 s stream, so the rate is $R(T)$ divided by the number of checkpoints.

## Deviation
A deviation is one specific "what if you acted differently here" alternative that you test your agent against. 
It is a rule, $d$, that takes in the agent's policy at time $t$, $\pi_t$, and gives back a different policy, $d(\pi_t)$, used from $t$ onwards.

You measure regret against a deviation (how much better $d(\pi_t)$ would have done compared to what $\pi_t$ did).



# Adaptive controller
The adaptive controller adapts to its environment by working out its own beleifs in real time.
It can adapt its guess for mass, friction, etc.

### Learning method
For the simple block push example, we define the push as:

    push = mass * acceleration + friction

This equation is of the form **y = mx + b**. 

Bearing that in mind, it is clear that if we were to plot out measured acceleration vs push, we could work out
both mass and friction. Friciton would be the y intercept, mass would be the slope.

This requires our controller to have memory of the past, naturally.

You can use Recursive Least Squares to update the fit incrementally rather than rerunning the fit for each new point. The update has the following shape:

    new estimate = old restimate + step size * (actual value - predicted value)

(This is the same shape as the incremental update rule in S&B 2.4)

### Forgetting
After friction changes, the old points still sit on the old line. They will bias your fit towards the earlier conditions for some time. To address this, we introduce a forgetting factor:

    λ = Forgetting Factor

Where at each step, every older points weigth is multiplied by λ. 

### Two limits to learning:
1. It only learns while the block slides. If acceleration is 0, you have no slope (accel = 0) so you just have that push = friction. This is not plottable and you can not infer additional information from this.

2. Acceleration needs to be varied. Similar concept, if acceleration is constant (same x value for all y values) then you can not draw a difinitively best fit line and again can not infer any information on mass or friction. 
If acceleration is not constant but not sufficiently varied, many lines will fit the 