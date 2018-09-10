import logging
import math
import os
import time
from collections import deque

import numpy as np
from tqdm import tqdm

from rl.util import Experience

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


def epsilon_decay_linear(epsilon_min, epsilon_max, i_episode, num_episodes):
    return epsilon_max - (epsilon_max - epsilon_min) * i_episode / float(num_episodes)


def epsilon_decay_exponential(epsilon_min, epsilon_max, i_episode, num_episodes):
    eps_decay = 0.995
    a = np.power(eps_decay, i_episode - 1)
    return max(epsilon_min, epsilon_max * np.power(eps_decay, i_episode - 1))


# https://github.com/udacity/deep-reinforcement-learning/blob/master/lab-taxi/monitor.py
class Simulator:

    @classmethod
    def train(self, env, agent, num_episodes=20000, window=100, epsilon_min=0.01,
              epsilon_max=1.0, max_steps=None, solved_score=None,
              render_every=None, progress=True, epsilon_decay_fn=epsilon_decay_exponential,
              **render_args):
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
        actual_scores = []
        avg_scores = deque(maxlen=num_episodes)
        # initialize best average reward
        best_avg_score = -math.inf
        # initialize monitor for most recent rewards
        scores_window = deque(maxlen=window)
        # for each episode
        epsilon = 1.
        solved_in_episodes = None
        pbar = tqdm(range(1, num_episodes + 1), disable=not progress)
        pbar.unit = 'episode'
        for i_episode in pbar:
            pbar.set_description("Training")
            # begin the episode
            state = env.reset()
            # initialize the sampled reward

            if (render_every is not None) and (i_episode % render_every == 0):
                env.render(**render_args)

            epsilon = epsilon_decay_fn(epsilon_min, epsilon_max, i_episode, num_episodes)

            experiences = []

            agent.on_episode_begin(state)

            score = 0
            i_step = -1
            while (max_steps is None) or (i_step < max_steps - 1):
                i_step += 1
                # agent selects an action
                action = agent.act(state, epsilon)
                # agent performs the selected action
                next_state, reward, done, _ = env.step(action)

                experience = Experience(state, action, reward, next_state, done)
                experiences.append(experience)

                if (render_every is not None) and (i_episode % render_every == 0):
                    env.render(**render_args)
                # agent performs internal updates based on sampled experience
                agent.step(experience)
                # update the sampled reward
                score += reward
                # update the state (s <- s') to next time step
                state = next_state
                if done:
                    break

            agent.on_episode_end(experiences)

            # save final sampled reward
            actual_scores.append(score)
            scores_window.append(score)

            avg_score = np.mean(scores_window)
            if i_episode >= window:
                # get average reward from last 100 episodes
                # append to deque
                avg_scores.append(avg_score)
                # update best average reward
                if avg_score > best_avg_score:
                    best_avg_score = avg_score

            progress_text = "ε: {:.4f}, Best: {:.2f}, Avg: {:.2f}, Steps: {:3d}".format(
                epsilon, best_avg_score, avg_score, i_step + 1)
            pbar.set_postfix_str(progress_text)

            if not progress:
                logger.debug("[Training %d/%d episodes] %s" % (i_episode, num_episodes, progress_text))

            # check if task is solved
            if solved_score is not None and best_avg_score >= solved_score:
                pbar.close()
                solved_in_episodes = i_episode
                agent.on_env_solved()
                break

        return actual_scores, best_avg_score, solved_in_episodes

    @classmethod
    def test(self, env, agent, num_episodes=200, max_steps=None, epsilon=0.005, **render_args):
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

            score = 0
            i_step = -1
            while (max_steps is None) or (i_step < max_steps - 1):
                i_step += 1
                action = agent.act(state, epsilon=epsilon)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                env.render(**render_args)
                score += reward

                if done:
                    break

            progress_text = "Score: {:.2f}, Steps: {:3d}".format(score, i_step + 1)
            pbar.set_postfix_str(progress_text)

            time.sleep(1)
