#!/usr/bin/env python3

import argparse
import logging
import os

from rl import Simulator
from rl.util import get_config_from_yaml
from train import create_env, create_agent, create_checkpoint_dir

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


def main(conf):
    env = create_env(conf, None, train_mode=False)
    agent = create_agent(conf, env)

    checkpoint_dir = create_checkpoint_dir(conf.name, conf.checkpoint_dir)

    model_name = conf.name

    files = []
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        files = os.listdir(checkpoint_dir)

    model_file = None
    if args.model_file and len(args.model_file) > 0:
        model_file = args.model_file[0]
        agent.load_model(model_file)
    else:
        load_from_checkpoints(agent, checkpoint_dir, files, model_name)

    Simulator.test(env, agent, num_episodes=conf.num_episodes)

    env.close()


def load_from_checkpoints(agent, checkpoint_dir, files, model_name):
    loaded = False
    for suffix in (".solved", "",):
        model_file = "%s%s" % (model_name, suffix)
        found = False
        for f in files:
            try:
                f.index(model_file)
                found = True
                break
            except:
                pass

        if not found:
            continue

        model_file = "%s/%s" % (checkpoint_dir, model_file)
        logger.info("Checkpoint file : %s" % model_file)
        model_file = agent.load_model(model_file)
        logger.info("Checkpoint loaded from : %s" % model_file)
        loaded = True
        break
    if not loaded:
        logger.warning("No checkpoint found, behaviour will be random.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-e', '--episodes', default=10, help='Number of episodes')
    parser.add_argument('-m', '--model_file', nargs=1, help='Model file to use.')
    args = parser.parse_args()

    conf = get_config_from_yaml(args.config)
    conf.name = os.path.basename(args.config).split(".")[0]
    conf.num_episodes = args.episodes

    main(conf)
