# SARSA Learning Algorithm


## AIM
To implement the SARSA (State–Action–Reward–State–Action) learning algorithm to find the optimal policy and value function for a given environment and compare its performance with the Monte Carlo method.

## PROBLEM STATEMENT
Reinforcement Learning aims to train an agent to make a sequence of decisions in an environment to maximize cumulative rewards.
In this experiment, the goal is to train an agent using the SARSA algorithm, an on-policy Temporal Difference (TD) method, where the agent learns the action-value function based on the policy it follows.
The environment (e.g., FrozenLake-v1 or gym-walk) provides discrete states and actions. The agent must learn the best policy to reach the goal while minimizing penalties or negative rewards..

## SARSA LEARNING ALGORITHM
1. **Initialize**
   - Initialize `Q(s, a)` arbitrarily for all state–action pairs.
   - Set parameters:
     - Learning rate `α`
     - Discount factor `γ`
     - Exploration rate `ε`

2. **For each episode:**
   - Initialize the starting state `s`.
   - Choose an action `a` from state `s` using an ε-greedy policy derived from `Q`.

3. **For each step of the episode:**
   - Take action `a`, observe **reward** `r` and **next state** `s′`.
   - Choose the next action `a′` from `s′` using ε-greedy(Q).
   - Update the action-value function:
     \[
     Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]
     \]
   - Set `s ← s′`, `a ← a′`.

4. **Repeat** until `s` is a terminal state.


## SARSA LEARNING FUNCTION
### Name: DHARSHINI K
### Register Number: 212223230047

```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: (np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state])))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

  
            state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT
### Optimal policy, optimal value function and success rate for the optimal policy

<img width="513" height="719" alt="image" src="https://github.com/user-attachments/assets/0835b6ed-8b04-4574-a4e0-d8209da21f26" />
<img width="542" height="407" alt="image" src="https://github.com/user-attachments/assets/af2a9c79-f9cf-4583-91f3-afdd4b397b42" />
<img width="830" height="723" alt="image" src="https://github.com/user-attachments/assets/fbf600d9-5872-42fb-bbbe-680426de0dfe" />
<img width="404" height="299" alt="image" src="https://github.com/user-attachments/assets/f88a1814-ee5a-49cc-a695-abecb7e1bdc9" />
<img width="873" height="724" alt="image" src="https://github.com/user-attachments/assets/4e38a24e-3db2-45aa-9929-4d31d2276417" />

### Plot comparing the state value functions of Monte Carlo method and SARSA learning.

<img width="1435" height="756" alt="image" src="https://github.com/user-attachments/assets/277b3735-e528-486c-8098-c6ab05891776" />
<img width="1434" height="756" alt="image" src="https://github.com/user-attachments/assets/ca36192c-6f41-4622-a202-d3f70039be1f" />

## RESULT
The SARSA algorithm successfully learned the optimal policy and value function for the given environment.
The learned policy closely approximated the optimal policy derived from Monte Carlo methods.
The comparison plot shows that SARSA achieves stable convergence with slightly more bias due to its on-policy nature.
