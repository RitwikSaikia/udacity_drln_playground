import argparse
import logging
import os

import matplotlib.pyplot as plt

from rl import GymEnv, Simulator, DqnAgent, RandomAgent, SarsaAgent, DqnModel, DuelingDqnModel, DqnConvModel, \
    get_backend, UnityEnv
from rl.util import get_config_from_yaml

ENV_TYPES = {
    'GymEnv': GymEnv,
    'UnityEnv': UnityEnv
}

AGENT_TYPES = {
    'RandomAgent': RandomAgent,
    'SarsaAgent': SarsaAgent,
    'DqnAgent': DqnAgent,
}

DQN_MODEL_TYPES = {
    'DqnModel': DqnModel,
    'DuelingDqnModel': DuelingDqnModel,
    'DqnConvModel': DqnConvModel,
}

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)


def _assert_in(key, values):
    if key not in values:
        raise Exception("Invalid key '%s', allowed values '%s'" % (key, values))


def _get_value_if_not(key):
    if not key:
        return None
    return key


def main(conf):
    if conf.headless:
        logger.info("Running in headless mode")

    env = create_env(conf)
    agent = create_agent(conf, env)

    max_steps = _get_value_if_not(conf.train.max_steps_per_episode)
    render_every = _get_value_if_not(conf.train.render_every)
    if conf.headless:
        render_every = None

    checkpoint_dir = create_checkpoint_dir(conf.name, conf.checkpoint_dir)

    logging.info("Checkpoint dir : %s" % checkpoint_dir)

    model_name = conf.name

    avg_rewards, best_avg_reward, solved_in_episodes = Simulator.train(env, agent,
                                                                       num_episodes=conf.train.num_episodes,
                                                                       max_steps=max_steps,
                                                                       solved_score=conf.env.solved_score,
                                                                       render_every=render_every)

    if solved_in_episodes:
        logging.info("Environment solved in %d episodes" % solved_in_episodes)

    suffix = ''
    if solved_in_episodes:
        suffix = '.solved'

    model_file = "%s%s" % (model_name, suffix)
    model_file = "%s/%s" % (checkpoint_dir, model_file)
    logging.info("Checkpoint file : %s" % model_file)
    model_file = agent.save_model(model_file)
    logging.info("Checkpoint saved to : %s" % model_file)

    if not args.headless:
        plt.plot(avg_rewards)
        plt.show()

    env.close()


def create_checkpoint_dir(name, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = "%s/%s" % (checkpoint_dir, name)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    return checkpoint_dir


def create_agent(conf, env):
    _assert_in(conf.agent.type, AGENT_TYPES.keys())
    agent_class = AGENT_TYPES[conf.agent.type]
    if conf.agent.type == 'DqnAgent':
        _assert_in(conf.agent.dqn.model, DQN_MODEL_TYPES.keys())
        model_class = DQN_MODEL_TYPES[conf.agent.dqn.model]

        def create_model_fn(input_shape, action_shape):
            return model_class(input_shape, action_shape)

        agent = agent_class(env, create_model_fn,
                            use_prioritized_experience_replay=conf.agent.dqn.use_prioritized_experience_replay,
                            use_double_dqn=conf.agent.dqn.use_double_dqn)
    else:
        agent = agent_class(env)
    return agent


def create_env(conf):
    _assert_in(conf.env.type, ENV_TYPES.keys())
    if conf.env.type == 'GymEnv':
        env = GymEnv(conf.env.gym.id)
    elif conf.env.type == 'UnityEnv':
        unity = conf.env.unity
        if unity.mode == 'vector':
            env = UnityEnv(unity.filename, unity.mode)
        elif unity.mode == 'visual':
            env = UnityEnv(unity.filename, unity.mode,
                           use_grayscale=unity.visual.use_grayscale,
                           frame_size=unity.visual.frame_size,
                           n_frames=unity.visual.num_frames)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, nargs=1, help='Config file')
    parser.add_argument('--headless', action='store_true', help='Disable rendering')
    parser.add_argument('-e', '--episodes', default=0, help='Override train.num_episodes')
    args = parser.parse_args()

    args.config = args.config[0]
    conf = get_config_from_yaml(args.config)
    conf.headless = args.headless
    conf.name = os.path.basename(args.config).split(".")[0]

    if int(args.episodes) > 0:
        conf.train.num_episodes = int(args.episodes)

    main(conf)
