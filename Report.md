# DRLND Project Navigation Report: Unity Banana Environment

## Summary

* Solved in 518 episodes.
* Algorithms implemented
   - [Deep Q Learning](#learning-algorithm)
   - [Dueling Deep Q Learning](#dueling-dqn-model)
   - [Double Deep Q Learning](#double-dqn)
   - [Prioritized Experience Replay](#prioritized-experience-replay)
* Trained Agent Playing

![Watch](reports/banana/2018-09-02.gif)

## [Learning Algorithm](#learning-algorithm)

1. Create a [PrioritizedReplayBuffer](#prioritized-experience-replay) to record every interaction the agent does as an `Experience(state, action, reward, next_state, done)`
2. Initialize two networks named `local` and `target`, of type [Dueling DQN](#dueling-dqn-model).
3. For every episode
    1. reset the environment, and record the state
    2. While environment is not done:
        1. Use ε-greedy method to get the action, use the `target` network for exploitation [(Double DQN)](#double-dqn).
        2. Interact with the environment with the above mentioned action
        3. Record the `Experience` to the `ReplayBuffer`
        4. After every 4th step, learn if we have minimum of `batch_size` experiences in buffer.
            1. Sample `batch_size` `Experiences` from the buffer (based on priority).
            2. Compute `Qsa_expected = rewards + gamma * max(target(next_states))`
            3. Compute `TD_error = mean-square-error(Qsa_expected - local(states)[actions])`
            4. Back propogate `TD_error` on the `local` network
            5. Copy the weights from `local` network to `target` network. ([Fixed Target Network](#fixed-target-network)).
        5. If `done`, terminate this episode.
    3. If average score of last 100 episodes >= solved_score, terminate training.

***

## Algorithm Details

## [Fixed Target Network](#fixed-target-network)

_Problem it solves:_ **Chasing a moving target.**

If we use a single network to train and to guess rewards, the behaviour will have huge oscillations.
Philosophically its better to observe the big picture, and add up the recent learning to our already attuned model.

`θ_target = τ * θ_local + (1 - τ) * θ_target`
 
`(τ = 0.001)`

The above equation here means, keep majority of our old learning from `θ_target` and add a very little portion of our recent learning `θ_local`.

_Reference_: [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

## [Dueling DQN Architecture](#dueling-dqn-model)

_Problem it solves:_ **Not every states are equally valuable to be considered.**

Environments has some states where a wrong action can impact the reward heavily, where as in some states,
no matter what action we take, it really doesn't matter.

The previous single stream DQN architecture was not able to address this issue. 

This architecture, whereas separates two different streams to representation of state values 
and action advantages. Combined back with an aggregation function.

![DuelingDQNModel](reports/resources/DuelingDqnModel.png)

```
fc1_units = 64
fc2_units = 32
fc3_units = 32
fc4_units = 32
```

_Reference_: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## [Prioritized Experience Replay](#prioritized-experience-replay)

_Problem it solves:_ **Not every Experience delivers an equally good lesson.**

A plain `ReplayBuffer` weighs every `Experience` equally while sampling. 
Not all `Experiences` are equal, and sometimes they are very rare.

This method addresses the issue, by sorting the `Experiences` by amount of `Error` that it generates during the learning process,
and then constructing a batch of all equally segmented priorities.

(The implementation also contains Importance Sampling as mentioned in the paper, 
but it seems to be buggy. Need to fix it.)

PS: An efficient implementation of the priority queue is taken from [SumTree](https://github.com/jaara/AI-blog/blob/master/SumTree.py)

_Reference_: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## [Double DQN Learning](#double-dqn)

_Problem it solves:_ **Overestimation of Q values.**

During learning, not every states and actions are explored equally, 
this may cause the network to generate favoritism towards highly explored ones.
This issue is more severe at the beginning of the learning process.

To solve this issue, rather than relying on one network, we use two networks.
1. The `local` network guesses the action, based on maximum action value.
2. The TD target is computed with a `target` network's action value estimate for above action.

![Double DQN](reports/resources/double_dqn.png)   

_Reference_: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

***

## Hyper Parameters
```
epsilon_min = 0.01
epsilon_decay = 0.995
optimizer = Adam(learning_rate = 5e-4)
gamma = 0.99
tau = 1e-3
batch_size = 64
buffer_size = int(1e5)
update_every = 4
```

## Results

The environment is solved in 518 episodes, with a score of 13.03

### Plot of Rewards
![Plot](reports/banana/2018-09-02.png)

### Logs

```
...
DEBUG:simulator.py:[Training 510/1000 episodes] ε: 0.0780, Best: 12.57, Avg: 12.57, Steps: 300
DEBUG:simulator.py:[Training 511/1000 episodes] ε: 0.0776, Best: 12.68, Avg: 12.68, Steps: 300
DEBUG:simulator.py:[Training 512/1000 episodes] ε: 0.0772, Best: 12.69, Avg: 12.69, Steps: 300
DEBUG:simulator.py:[Training 513/1000 episodes] ε: 0.0768, Best: 12.69, Avg: 12.69, Steps: 300
DEBUG:simulator.py:[Training 514/1000 episodes] ε: 0.0764, Best: 12.74, Avg: 12.74, Steps: 300
DEBUG:simulator.py:[Training 515/1000 episodes] ε: 0.0760, Best: 12.74, Avg: 12.73, Steps: 300
DEBUG:simulator.py:[Training 516/1000 episodes] ε: 0.0757, Best: 12.76, Avg: 12.76, Steps: 300
DEBUG:simulator.py:[Training 517/1000 episodes] ε: 0.0753, Best: 12.87, Avg: 12.87, Steps: 300
DEBUG:simulator.py:[Training 518/1000 episodes] ε: 0.0749, Best: 13.03, Avg: 13.03, Steps: 300
INFO:root:Environment solved in 518 episodes
```

## Ideas for Future Work

1. Compare the effects of various algorithms on learning speed.
2. Use `DuelingDqnCNNModel` to solve VisualBanana problem. Learning from seeing whats visible in terms of pixels.
3. Try out various Atari like visual environments.
4. Apply improved algorithms.