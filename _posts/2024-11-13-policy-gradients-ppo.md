An emprirically successful reinforcement learning algorithm that generalizes well to several environments in single agent and multi-agent settings is Proximal Policy Optimization, PPO for short or MAPPO in multi-agent scenarios (cite PPO paper). PPO is part of the gradient policy family of algorithms. Here is a short list of some advantages of PPO:
- Applicable to continuous control tasks and tasks with discrete action space
- Naturally explores action spaces without needing to include a noise process or entropy term
- Robust to noise and under the default hyperparameter settings

Now there already exists an abundance of great resources on PPO, for example, OpenAI's spinning up documentation and this ICML blog post (link to sites). What this post aims to do is elucidate the theory and proof behind gradient policy from (cite Sutton paper) to understand why they work. I will also include an annotated implementation of PPO in Python at the end. So if you are really only interested in seeing the code, then you can skip all the way to the bottom. I will only be focusing on the single agent setting.

In what follows I will reformulate the single agent reinforcement learning setting, regurgitate the motivation for policy gradient methods, restate the policy gradient theorem, expound on the proof, and add additional analysis of policy gradient from (cite levin slides).

At the center of reinforcement learning is a Markov Decision Process (MDP) which can be described by a tuple $(S, A, R, P)$. Here $S$ and $A$ are the space of all possible states that an agent can arrive at and actions that the agent can take. $R$ is the reward function, $R: S x A -> \R$, in which you input states and actions and receive a scalar. Then $P: S x A x S' -> $ is the transition function describing the probability of arriving at state s' when the agent is in state s and takes action a. MDPs can be episodic aka finite horizon or infinite horizon. In the latter case, there must be a discount factor $\gamma$ to ensure convergence of any solutions to the MDP.

There are two overarching paradigms to solving reinforcement learning problems, value iteration and policy iteration. A famous example of a value iteration algorithm is Q-Learning. As you may have guessed policy gradient methods and PPO fall under the policy iteration paradigm. Please refer to Sutton & Barto 2012 for more details on policy and value iteration. 


 

