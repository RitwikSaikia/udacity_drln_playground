import sys

sys.path.append("../")

import argparse

import matplotlib.pyplot as plt
import numpy as np
from rl import GymEnv, SarsaAgent, Simulator


def main(args):
    env = GymEnv("Taxi-v2")
    agent = SarsaAgent(env)

    if not args.skip_train:
        avg_rewards, best_avg_reward = Simulator.train(env, agent, solved_avg_reward=9.7)

        plt.plot(avg_rewards)
        plt.ylim([min(0, np.max(avg_rewards)), 10])
        plt.show()

    if not args.headless and not args.skip_test:
        Simulator.test(env, agent, num_episodes=1, mode='human')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGym Taxi Demo')
    parser.add_argument('--skip_train', action='store_true', help='Skip training')
    parser.add_argument('--skip_test', action='store_true', help='Skip testing')
    parser.add_argument('--headless', action='store_true', help='Disable rendering')
    args = parser.parse_args()

    main(args)
