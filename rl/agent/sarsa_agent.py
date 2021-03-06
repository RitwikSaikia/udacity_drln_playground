import pickle
import random
from collections import defaultdict

import numpy as np

from .agent import _AbstractAgent


class SarsaAgent(_AbstractAgent):

    def __init__(self, env, alpha=0.1, gamma=1.0, mode='exp'):
        super().__init__(env)
        allowed_modes = ('max', 'exp')
        if mode not in allowed_modes:
            raise Exception("Invalid mode '%s', allowed values = '%s'" % (mode, allowed_modes))
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.visited_states = {}
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode

    def act(self, state, epsilon=0.01):
        state = np.argmax(state)
        self.visited_states[state] = True
        if random.random() > epsilon:
            return np.argmax(self.Q[state])
        return self.env.sample_action()

    def step(self, experience):
        state, action, reward, next_state, done = experience
        state = np.argmax(state)
        next_state = np.argmax(next_state)
        self.visited_states[state] = True
        if self.mode == 'max':
            Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        elif self.mode == 'exp':
            policy_s = self.get_policy(next_state)
            Qsa_next = np.dot(self.Q[next_state], policy_s)
        else:
            raise AssertionError()

        self.Q[state][action] = self.update_Q(self.Q[state][action],
                                              Qsa_next,
                                              reward, self.alpha, self.gamma)

    def get_policy(self, state, epsilon=0.05):
        state = np.argmax(state)
        Qs = self.Q[state]
        policy_s = np.zeros(self.nA) + epsilon / self.nA
        max_value = np.max(Qs)
        max_mask = np.argwhere(Qs == max_value)
        if len(max_mask) > 1:
            max_idx = np.random.choice(max_mask[0])
        else:
            max_idx = max_mask
        policy_s[max_idx] += (1 - epsilon)
        return policy_s

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)

    def save_model(self, filepath):
        filepath += self.get_model_ext()
        Q = {}
        for state in self.visited_states.keys():
            Q[state] = self.Q[state]
        pickle.dump(Q, open(filepath, "wb"))
        return filepath

    def load_model(self, filepath):
        filepath += self.get_model_ext()
        Q = pickle.load(open(filepath, "rb"))
        for state in Q.keys():
            self.Q[state] = Q[state]
        return filepath

    def get_model_ext(self):
        return ".pkl"



