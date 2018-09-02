import random
from abc import abstractmethod
from collections import deque, namedtuple

import numpy as np


class _AbstractReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, k):
        raise NotImplementedError()

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class ReplayBuffer(_AbstractReplayBuffer):

    def __init__(self, capacity):
        super().__init__(capacity)
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def remember(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, k):
        experiences = random.sample(self.memory, k=k)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return {}, (states, actions, rewards, next_states, dones)

    def update(self, **kwargs):
        pass

    def __len__(self):
        return len(self.memory)
