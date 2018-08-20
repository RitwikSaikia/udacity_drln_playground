import argparse

import gym
import matplotlib.pyplot as plt
from dask.tests.test_base import np

from rl import Engine
from rl.agent import SarsaAgent


def main(args):
    env = gym.make("Taxi-v2")
    agent = SarsaAgent(env)

    if not args.skip_train:
        avg_rewards, best_avg_reward = Engine.train(env, agent, solved_avg_reward=9.7)

        plt.plot(avg_rewards)
        plt.ylim([min(0, np.max(avg_rewards)), 10])
        plt.show()

    if not args.skip_test:
        Engine.test(env, agent, num_episodes=1, mode='human')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGym Taxi Demo')
    parser.add_argument('--skip_train', action='store_true', help='Skip training.')
    parser.add_argument('--skip_test', action='store_true', help='Skip training.')
    args = parser.parse_args()

    main(args)
