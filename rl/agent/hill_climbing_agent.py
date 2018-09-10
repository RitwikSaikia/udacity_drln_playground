import pickle

import numpy as np

from rl.agent.agent import _AbstractAgent


class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs)  # option 1: stochastic policy
        action = np.argmax(probs)                # option 2: deterministic policy
        return action


class HillClimbingAgent(_AbstractAgent):

    def __init__(self, env, gamma=1.0, noise_scale=1e-2):
        super().__init__(env)

        self.policy = Policy()
        self.gamma = gamma
        self.best_R = -np.Inf
        self.best_w = self.policy.w
        self.noise_scale = noise_scale

    def act(self, state, epsilon=0.01):
        return self.policy.act(state)

    def on_episode_begin(self, state):
        pass

    def on_episode_end(self, experiences):
        rewards = [e.reward for e in experiences]
        n_steps = len(rewards)

        discounts = [self.gamma ** i for i in range(n_steps + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= self.best_R:
            self.best_R = R
            self.best_w = self.policy.w
            self.noise_scale = max(1e-3, self.noise_scale / 2)
            self.policy.w += self.noise_scale * np.random.rand(*self.policy.w.shape)
        else:
            self.noise_scale = min(2, self.noise_scale * 2)
            self.policy.w = self.best_w + self.noise_scale * np.random.rand(*self.policy.w.shape)

    def on_env_solved(self):
        self.policy.w = self.best_w

    def save_model(self, filename):
        filename += ".pkl"
        pickle.dump(self.policy.w, open(filename, "wb"))
        return filename

    def load_model(self, filename):
        filename += ".pkl"
        w = pickle.load(open(filename, "rb"))
        self.policy.w = w
        return filename
