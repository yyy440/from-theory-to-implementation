(**Under Construction**)
# PPO Implementation and Understanding the Theory Behind Policy Gradients
---
## Introduction
---
An emprirically successful reinforcement learning algorithm that generalizes well to several environments in single agent and multi-agent settings is Proximal Policy Optimization, PPO for short or MAPPO in multi-agent scenarios (cite PPO paper). PPO is part of the gradient policy family of algorithms. Here is a short list of some advantages of PPO:
- Applicable to continuous control tasks and tasks with discrete action space
- Naturally explores action spaces without needing to include a noise process or entropy term
- Robust to noise and under the default hyperparameter settings

Now there already exists an abundance of great resources on PPO, for example, OpenAI's spinning up documentation and this ICML blog post (link to sites). What this post aims to do is elucidate the theory and proof behind gradient policy from (cite Sutton paper) to understand why they work. I will also include an annotated implementation of PPO in Python at the end. So if you are really only interested in seeing the code, then you can skip all the way to the bottom. I will only be focusing on the single agent setting.

In what follows I will reformulate the single agent reinforcement learning setting, regurgitate the motivation for policy gradient methods, restate the policy gradient theorem, expound on the proof, and add additional analysis of policy gradient from (cite levin slides). I will be following the notation used in the paper to avoid confusion. This blog is not for a complete beginner to RL. The reader should know the definitions of at least the value functions, Bellman equations, policy/value iteration, general policy improvement, and calculus.
## RL Background
---
At the center of reinforcement learning is a Markov Decision Process (MDP) which can be described by a tuple $(S, A, R, P)$. Here $S$ and $A$ are the space of all possible states that an agent can arrive at and actions that the agent can take. $R$ is the reward function, $R: S \times A \rightarrow \mathbb{R}$, in which you input states and actions and receive a scalar. Then $P: S \times A \times S' \rightarrow \[0,1\]$ is the transition function describing the probability of arriving at state s' when the agent is in state s and takes action a. As the state transitions are probabilistic, the rewards are then defined in expectation, $R_{s}^{a} = \mathbb{E}\[r_{t+1}|s_{t}, a_{t}\]$; depending on which state the agent acutally ends up in is what reward they would actually recieve which is set out by the environment. For analysis it is necessary to use expected rewards (in case of deterministic environment, $P_{ss'}^{a}=1$). Lastly, MDPs can be episodic aka finite horizon or infinite horizon. In the latter case, there must be a discount factor $\gamma$ to ensure convergence of any solutions to the MDP. 

Having laid out the foundations, there are two ways to express the goal of reinforcement learning. One is to maximize expected rewards:

$\rho (\pi) = lim_{n\rightarrow\infty} \frac{1}{n}\mathbb{E}\[r_{1} + r\{2} + ... + r_{n}|\pi] = \sum_{s}d^{\pi}(s)\sum_{a}\pi(s,a)R_{s}^{a}$

(Skip this paragraph if you understand the equation) To break it down in words we can use $\rho$ which is a function of policies to rank how desirable a policy is. Here $\rho$ is the average of the expected sum of the rewards when using policy $\pi$. Expanding a little more, remember in an environment you start at a random state $s_{0}$ choose actions based on the policy and get a reward based on the action chosen, then move to a new state $s_{1}$ and repeat until termination. This process generates a sequence rewards $r_{1}, r_{2}, ...$. Starting in partialerent states may lead to a partialerent sequence of rewards, so each reward sequence is seen with a certain probability and so each reward series is mutliplied by that probability which is where the expectation comes in. The rightmost expression is $\rho$ and the middle equation in a partialerent form which comes in handy later. This introduces d^{\pi}(s) which is the stationary distribution of state s. Here is equation mathematically, $d^{\pi} = lim_{t->\infty} P(s_{t}=s|s_{0},\pi)$. In words, when you follow a policy for an infinite amount of time and count up the number of times each state is visited, you get the percentage or probability that state is visited when using this specific policy. Since the agent is going for infinite time, this probability is independent of whatever state you may start in as you will start in every state and infinite amount of times and the probability of visiting state s under each start state will converge. 

Having $\rho(\pi)$, we can get the state-action value which is:

$Q^{\pi}(s,a) = \sum_{t=0}^{\infty}\mathbb{E}\[r_{t} - \rho{\pi}|s_{0}=s,a_{0}=a,\pi\]$

Recall that the state-action value gives a score to how good it is to be in a certain state and taking a specific action.

The second view, is the same as is introduced in (Sutton & Barto 2018), a discounted view:

$\rho(\pi) = \mathbb{E}\[\sum_{t=1}^{\infty}\gamma^{t-1}r_{t}|s_{0}, \pi\]$
and 
$Q^{\pi}(s,a) = \mathbb{E}\[\sum_{k=1}^{\infty}\gamma^{k-1}r_{t+k}|s_{t}=s,a_{t}=a,\pi\]$ (The expectation in both equations is taken w.r.t the policy).
In addition, the steady state distribution of s is now $d^{\pi}(s) = \sum_{t=0}^{\infty}\gamma^{t}P(s_{t}=s|s_{0},\pi)$. In words, the frequnecy of state visits to a particular state s is the discounted sum of the probability that state is visited when starting in state s_{0} and following policy $\pi$, at every timestep. Exponential discounting bounds the value so that it is finite. 

## Policy Gradient Theorem
---
At this point, the paper introduces the Policy Gradient Theorem:
For any MDPs, <br />

$\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi}{\partial\theta}Q^{\pi}(s,a)$. <br />
($\theta$ are the parameters of the policy) <br />
**The most important takeaway is that the steady state distribution does not change with changes to policy parameters since no $\frac{\partial d^{\pi}}{\partial\theta}$ is present.** Suppose it were the case that the state distribution changed as the policy is changed, then this affects $\rho$ which in turn, changes $Q^{\pi}$ leading to biased estimates ($Q^{\pi}$ can be approximated by returns)......
### Proof
Here I will walk through each line of the the proof adding comments on how it came to be, which mostly uses the definitions of the above formulations. (Note: I will be liberal in leaving out parts of notation, because it takes more time to type and takes up more space.) <br />
To start (expanding this out will give us $\frac{\partial\rho}{\partial\theta}$): <br />
$\frac{\partial V^{\pi}(s)}{\partial\theta} = \frac{\partial}{\partial\theta}\sum_{a}\pi(s,a)Q^{\pi}(s,a)$ <br />

Using the rules of calculus, we can take the derivative into the sum and apply the product rule: <br />
$\frac{\partial V^{\pi}(s)}{\partial\theta} = \sum_{a}(\frac{\partial\pi}{\partial\theta}Q^{\pi} + \pi \frac{\partial Q^{\pi}}{\partial\theta})$ <br />

Next it is beneficial to expand out the definition of $Q^{\pi}$ in the term to the right of the $+$ sign: <br />
$\frac{\partial V^{\pi}(s)}{\partial\theta} = \sum_{a}(\frac{\partial\pi}{\partial\theta}Q^{\pi} + \pi*\frac{\partial}{\partial\theta}(R_{s}^{a} - \rho(\pi) + \sum_{s'}(P_{ss'}^{a}V^{\pi}(s'))))$
<br />
Now pass the derivative operator through. Only $V^{\pi}$ depends on the policy params $\theta$ and finally $\partial{\rho}{\theta}$ appears: <br />
$\frac{\partial V^{\pi}(s)}{\partial\theta} = \sum_{a}(\frac{\partial\pi}{\partial\theta}Q^{\pi} + \pi*(-\frac{\partial\rho}{\partial\theta} + \sum_{s'}P_{ss'}^{a}\frac{\partial V^{\pi}}{\theta}))$ <br />

Rearranging terms, and we start to see the policy gradient theorem: <br />
$\frac{\partial\rho}{\partial\theta} = \sum_{a}(\frac{\partial\pi}{\partial\theta}*Q + \pi * \sum_{s'}P_{ss'}^{a} \frac{\partial V(s')}{\partial\theta} ) - \frac{\partial V(s)}{\partial\theta}$ <br />

One last piece to add is summing over $d^{\pi}$:  <br />
$\sum_{s}d^{\pi}(s)\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s) \sum_{a}\frac{\partial\pi}{\partial\theta}*Q + \sum_{s}d^{\pi}(s) \sum_{a}\pi \sum_{s'}P_{ss'}^{a} \frac{\partial V(s')}{\partial\theta} - \sum_{s}d^{\pi}(s) \frac{\partial V(s)}{\partial\theta}$ <br />

Leveraging the fact that $d^{\pi}$ is stationary: <br />
$\sum_{s}d^{\pi}(s)\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s) \sum_{a}\frac{\partial\pi}{\partial\theta}*Q + \sum_{s'}d^{\pi}(s')\frac{\partial V(s')}{\partial\theta} - \sum_{s}d^{\pi}(s) \frac{\partial V(s)}{\partial\theta}$ <br />
I would like to touch on the jump from $\sum_{s}d^{\pi}(s) \sum_{a}\pi \sum_{s'}P_{ss'}^{a} \frac{\partial V(s')}{\partial\theta}$ to $\sum_{s'}d^{\pi}(s')\frac{\partial V(s')}{\partial\theta}$. <br />
Qualitatively, start with the end point, which is $d^{\pi}(s')$ or the steady state distribution of the next states and the probability of getting to state $s'$ is dependent on the steady state probability of the current state $d^{\pi}(s)$, the policy $\pi(s,a)$, and the transition probabilities $P_{ss'}^{a}$.

The ultimate step to arrive at the policy gradient theorem is to notice $d^{\pi}$ sums to 1 as it is a proper probability distribution and $\sum_{s'}d^{\pi}(s')\frac{\partial V(s')}{\partial\theta} - \sum_{s}d^{\pi}(s) \frac{\partial V(s)}{\partial\theta} = 0$ since we are summing over all states in the left and right terms: <br />

$\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi}{\partial\theta}Q^{\pi}(s,a)$ <br />
The summation over the steady state distributions and derivative of state value functions cancel each other out as in both cases they are summing over all states (See appendix for a walk through of the proof in the second formulation).

The requirement is that we have a good approximate for $Q^{\pi}$. Following the paper's notation, let $f_{w}$ be an approximation for $Q^{\pi}$ where $f$ is a function parameterized by weights $w$. To learn $f_{w}$ it is possible to take actions according to $\pi$ and use the update rule: $\frac{\partial}{\partial w}\[\hat{Q}-f_{w}\]^{2}$. To take the derivative of this we need the chain rule, and that leaves us with: $\[\hat{Q}-f_{w}\]\frac{\partial f_{w}}{\partial w}$ 

