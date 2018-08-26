import random
from abc import abstractmethod

import gym
from unityagents import UnityEnvironment


class Env:
    action_shape = None
    state_shape = None
    action_space = None
    state_space = None

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        return NotImplementedError()

    @abstractmethod
    def sample_action(self):
        return random.choice(range(self.action_shape[0]))


class GymEnv(Env):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.env = gym.make(name)
        self.action_shape = self._extract_shape(self.env.action_space)
        self.state_shape = self._extract_shape(self.env.observation_space)

        self.action_space = (self.env.action_space.n,)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        self.env.close()

    def _extract_shape(self, space):
        state_type = type(space).__name__
        if state_type == 'Discrete':
            return space.n,
        return space.shape


class UnityEnv(Env):

    def __init__(self, name: str, filename: str, **kwargs) -> None:
        super().__init__(name)
        self.env = UnityEnvironment(filename, **kwargs)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.cur_env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.action_shape = (brain.vector_action_space_size,)
        self.state_shape = (len(self.cur_env_info.vector_observations[0]),)
        self.action_space = self.action_shape

    def reset(self):
        self.cur_env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self._to_state(self.cur_env_info)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = self._to_state(env_info)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.cur_env_info = env_info

        return next_state, reward, done, env_info

    def render(self, **kwargs):
        pass

    def close(self):
        pass

    def _to_state(self, env_info):
        return env_info.vector_observations[0]
