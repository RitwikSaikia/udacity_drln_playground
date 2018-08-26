import random
from collections import namedtuple

import numpy as np

from rl.util.priority_queue import PriorityQueue


class PrioritizedReplayBuffer:
    eps = 0.01
    alpha = 0.6
    beta = 0.4
    beta_annealing_delta = 0.001

    max_error = 1

    def __init__(self, buffer_size: int):
        self.queue = PriorityQueue(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def remember(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        max_score = self.queue.max_priority
        if max_score == 0:
            max_score = self.max_error
        self.queue.add(max_score, e)

    def sample(self, k: int):
        N = len(self)

        # Annealing the Bias
        self.beta = min(1, self.beta + self.beta_annealing_delta)

        sigma_p = self.queue.total_priority  # Σp^α

        # k segments of equal probability
        segment_size = sigma_p / k
        segment_priorities = map(lambda j: random.uniform(segment_size * j, segment_size * (j + 1)), range(k))

        # sample for each segment
        indexes, p, experiences = zip(*map(self.queue.get, segment_priorities))

        # P = p^α / Σp^α
        P = np.asarray(p) / sigma_p

        # Importance Sampling Weights
        # (N · P (j))^−β / max(w)
        p_max = self.queue.max_priority / sigma_p + np.finfo(float).eps
        max_weight = (N * p_max) ** (-self.beta)
        weights_is = (N * P) ** (-self.beta) / max_weight

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return (indexes, weights_is), (states, actions, rewards, next_states, dones)

    def update(self, indexes, delta):
        delta += self.eps  # p = |𝛿| + ε
        delta = np.minimum(delta, self.max_error)  # Clip to max_error
        priorities = np.power(delta, self.alpha)  # p^α

        for i, p in zip(indexes, priorities):
            self.queue.update(i, p)

    def __len__(self):
        return len(self.queue)
