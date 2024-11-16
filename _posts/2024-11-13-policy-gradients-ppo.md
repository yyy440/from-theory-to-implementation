An emprirically successful reinforcement learning algorithm that generalizes well to several environments in single agent and multi-agent settings is Proximal Policy Optimization, PPO for short or MAPPO in multi-agent scenarios (cite PPO paper). PPO is part of the gradient policy family of algorithms. Here is a short list of some advantages of PPO:
- Applicable to continuous control tasks and tasks with discrete action space
- Naturally explores action spaces without needing to include a noise process or entropy term
- Robust to noise and under the default hyperparameter settings

Now there already exists an abundance of great resources on PPO, for example, OpenAI's spinning up documentation and this ICML blog post (link to sites). What this post aims to do is elucidate the theory and proof behind gradient policy from (cite Sutton paper) to understand why they work. I will also include an annotated implementation of PPO in Python at the end. So if you are really only interested in seeing the code, then you can skip all the way to the bottom. I will only be focusing on the single agent setting.

In what follows I will reformulate the single agent reinforcement learning setting, regurgitate the motivation for policy gradient methods, restate the policy gradient theorem, expound on the proof, and add additional analysis of policy gradient from (cite levin slides).

At the center of reinforcement learning is a Markov Decision Process (MDP) which can be described by a tuple $(S, A, R, P)$. Here $S$ and $A$ are the space of all possible states that an agent can arrive at and actions that the agent can take. $R$ is the reward function, $R: S x A -> \R$, in which you input states and actions and receive a scalar. Then $P: S x A x S' -> $ is the transition function describing the probability of arriving at state s' when the agent is in state s and takes action a. MDPs can be episodic aka finite horizon or infinite horizon. In the latter case, there must be a discount factor $\gamma$ to ensure convergence of any solutions to the MDP. There are two ways to express the goal of reinforcement learning. One is to maximize expected rewards:

$\rho(\pi) = lim_{n->\infty} \frac{1}{n}\mathbb{E}\[r_{1} + r\{2} + ... + r_{n}|\pi] = \sum_{s}d^{\pi}(s)\sum_{a}\pi(s,a)R$

(Skip this paragraph if you understand the equation) To break it down in words we can use $\rho$ which is a function of policies to rank how desirable a policy is. Here $\rho$ is the average of the expected sum of the rewards when using policy $\pi$. Expanding a little more, remember in an environment you start at a random state $s_{0}$ choose actions based on the policy and get a reward based on the action chosen, then move to a new state $s_{1}$ and repeat until termination. This process generates a sequence rewards $r_{1}, r_{2}, ...$. Starting in different states may lead to a different sequence of rewards, so each reward sequence is seen with a certain probability and so each reward series is mutliplied by that probability which is where the expectation comes in. The rightmost expression is $\rho$ and the middle equation in a different form which comes in handy later. This introduces d^{\pi}(s) which is the stationary distribution of state s. Here is equation mathematically, $d^{\pi} = lim_{t->\infty} P(s_{t}=s|s_{0},\pi)$. In words, when you follow a policy for an infinite amount of time and count up the number of times each state is visited, you get the percentage or probability that state is visited when using this specific policy. Since the agent is going for infinite time, this probability is independent of whatever state you may start in as you will start in every state and infinite amount of times and the probability of visiting state s under each start state will converge. 

Having $\rho(\pi)$, we can get the state-action value which is:

$Q^{\pi}(s,a) = \sum_{t=0}^{\infty}\mathbb{E}\[r_{t} - \rho{\pi}|s_{0}=s,a_{0}=a,\pi\]$

Recall that the state-action value gives a score to how good it is to be in a certain state and taking a specific action.

In the second view, 

There are two overarching paradigms to solving reinforcement learning problems, value iteration and policy iteration. A famous example of a value iteration algorithm is Q-Learning. As you may have guessed policy gradient methods and PPO fall under the policy iteration paradigm. Essentially value iteration initializes random values for each state or state-action pair and iteratively updates the values by using Bellman optimality equations as update rules until the values converge. After convergence a greedy policy is paired with the value function to make decisions at each state. On the other hand in policy iteration, the policy is updated using a value function until the policy converges. Please refer to Sutton & Barto 2012 for more details on policy and value iteration. 


 

