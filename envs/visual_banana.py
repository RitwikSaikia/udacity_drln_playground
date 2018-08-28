import sys

sys.path.append("../")

import argparse
import os

import matplotlib.pyplot as plt

from rl import Simulator, UnityEnv, DqnAgent, DqnConvModel


def main(args):
    env = UnityEnv("VisualBanana", "VisualBanana.app", mode='visual')
    model_file = "visual_banana.model.h5"

    def create_model_fn(input_shape, action_shape):
        return DqnConvModel(input_shape, action_shape)

    agent = DqnAgent(env, create_model_fn)

    if not args.skip_train:
        render_every = 100 if not args.headless else None
        avg_rewards, best_avg_reward = Simulator.train(env, agent,
                                                       num_episodes=1000,
                                                       solved_avg_reward=13.0,
                                                       render_every=render_every)

        plt.plot(avg_rewards)
        plt.show()

        agent.save_model(model_file)

    if not args.headless and not args.skip_test:
        if os.path.exists(model_file):
            agent.load_model(model_file)
        else:
            print("Warning: Model file not found: %s" % model_file)

        Simulator.test(env, agent, num_episodes=10, max_steps=None, mode='human')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGym CartPole Demo')
    parser.add_argument('--skip_train', action='store_true', help='Skip training')
    parser.add_argument('--skip_test', action='store_true', help='Skip testing')
    parser.add_argument('--headless', action='store_true', help='Disable rendering')
    args = parser.parse_args()

    main(args)
