import argparse
import os

import gym
import matplotlib.pyplot as plt
from keras import Sequential
from keras.activations import relu
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from rl import Engine
from rl.agent import DqnAgent


def create_model(action_shape, state_shape, learning_rate):
    model = Sequential()
    model.add(Dense(64, input_shape=state_shape, activation=relu))
    model.add(Dense(64, activation=relu))
    model.add(Dense(action_shape))
    model.compile(loss=mean_squared_error, optimizer=Adam(lr=learning_rate))
    return model


def main(args):
    env = gym.make("CartPole-v1")
    model_file = "env_cart_pole.model.h5"
    max_steps = 500

    agent = DqnAgent(env, create_model)

    if not args.skip_train:
        render_every = 100 if not args.headless else None
        avg_rewards, best_avg_reward = Engine.train(env, agent,
                                                    num_episodes=2000,
                                                    max_steps=max_steps,
                                                    solved_avg_reward=195.0,
                                                    render_every=render_every)

        plt.plot(avg_rewards)
        plt.show()

        agent.save_model(model_file)

    if not args.headless and not args.skip_test:
        if os.path.exists(model_file):
            agent.load_model(model_file)
        else:
            print("Warning: Model file not found: %s" % model_file)

        Engine.test(env, agent, num_episodes=10, max_steps=None, mode='human')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGym CartPole Demo')
    parser.add_argument('--skip_train', action='store_true', help='Skip training')
    parser.add_argument('--skip_test', action='store_true', help='Skip testing')
    parser.add_argument('--headless', action='store_true', help='Disable rendering')
    args = parser.parse_args()

    main(args)
