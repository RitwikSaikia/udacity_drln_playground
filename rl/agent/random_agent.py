from .agent import _AbstractAgent


class RandomAgent(_AbstractAgent):

    def __init__(self, env):
        super().__init__(env)

    def act(self, state, epsilon=0.01):
        return self.env.sample_action()

    def step(self, state, action, reward, next_state, done):
        pass
