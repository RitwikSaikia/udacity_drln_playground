import random
from abc import abstractmethod
from collections import deque

import numpy as np

from .experience import Experience


class _AbstractReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def remember(self, experience: Experience):
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

    def remember(self, experience):
        self.memory.append(experience)

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
