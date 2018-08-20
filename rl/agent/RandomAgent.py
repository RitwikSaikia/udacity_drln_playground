from rl.agent import Agent


class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def act(self, state, epsilon=0.01):
        return self.env.action_space.sample()

    def step(self, state, action, reward, next_state, done):
        pass
