import random
from abc import abstractmethod

import gym


class Env:
    action_shape = None
    state_shape = None
    action_space = None
    state_space = None

    def __init__(self, name) -> None:
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

    def __init__(self, name) -> None:
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
