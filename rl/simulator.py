import math
import time
from collections import deque

import numpy as np
from tqdm import tqdm


class Simulator:

    @classmethod
    def train(self, env, agent, num_episodes=20000, window=100, epsilon_min=0.01,
              epsilon_decay=0.995, max_steps=None, solved_score=None,
              render_every=None, **render_args):
        """ Train

        :param env: instance of environment
        :param agent: instance of class Agent (see agent.py for details)
        :param num_episodes: number of episodes of agent-environment interaction
        :param window: number of episodes to consider when calculating average rewards
        :param epsilon_min: minimum value for exploration
        :param epsilon_decay: decay factor for exploration
        :param max_steps: max number of steps per episode, None means wait till done.
        :param solved_score: consider it solved, when best avg exceeds this
        :param render_every: render the environment after these many episodes
        :param render_args: additional parameters to be passed to env.render()
        :return:
            - avg_rewards: deque containing average rewards
            - best_avg_reward: largest value in the avg_rewards deque
        """
        # initialize average rewards
        avg_rewards = deque(maxlen=num_episodes)
        # initialize best average reward
        best_avg_reward = -math.inf
        # initialize monitor for most recent rewards
        rewards_window = deque(maxlen=window)
        # for each episode
        epsilon = 1.
        solved_in_episodes = None
        pbar = tqdm(range(1, num_episodes + 1))
        pbar.unit = 'episode'
        for i_episode in pbar:
            pbar.set_description("Training")
            # begin the episode
            state = env.reset()
            # initialize the sampled reward

            if (render_every is not None) and (i_episode % render_every == 0):
                env.render(**render_args)

            samp_reward = 0
            i_step = -1
            while (max_steps is None) or (i_step < max_steps - 1):
                i_step += 1
                # agent selects an action
                action = agent.act(state, epsilon)
                # agent performs the selected action
                next_state, reward, done, _ = env.step(action)

                if (render_every is not None) and (i_episode % render_every == 0):
                    env.render(**render_args)
                # agent performs internal updates based on sampled experience
                agent.step(state, action, reward, next_state, done)
                # update the sampled reward
                samp_reward += reward
                # update the state (s <- s') to next time step
                state = next_state
                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # save final sampled reward
            rewards_window.append(samp_reward)

            avg_reward = best_avg_reward
            if i_episode >= window:
                # get average reward from last 100 episodes
                avg_reward = np.mean(rewards_window)
                # append to deque
                avg_rewards.append(avg_reward)
                # update best average reward
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
            pbar.set_postfix_str("Best: {:.2f}, Avg: {:.2f}, ε: {:.4f}"
                                 .format(best_avg_reward, avg_reward, epsilon))

            # check if task is solved
            if solved_score is not None and best_avg_reward >= solved_score:
                pbar.close()
                solved_in_episodes = i_episode
                break

        return avg_rewards, best_avg_reward, solved_in_episodes

    @classmethod
    def test(self, env, agent, num_episodes=200, max_steps=None, **render_args):
        """
        Test

        :param env: instance of environment
        :param agent: instance of class Agent (see agent.py for details)
        :param num_episodes: number of episodes of agent-environment interaction
        :param max_steps: max number of steps per episode, None means wait till done.
        :param render_args: additional parameters to be passed to env.render()
        :return:
        """
        pbar = tqdm(range(1, num_episodes + 1))
        pbar.unit = 'episode'
        for i_episode in pbar:
            pbar.set_description("Testing")
            state = env.reset()
            env.render(**render_args)

            i_step = -1
            while (max_steps is None) or (i_step < max_steps - 1):
                i_step += 1
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                env.render(**render_args)

                if done:
                    break

            time.sleep(2)
