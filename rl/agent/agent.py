class _AbstractAgent:

    def __init__(self, env):
        """ Initialize agent.

        Params
        ======
        - env: the environment, agent is supposed to interact with
        """
        self.env = env

        self.action_shape = self.env.action_shape
        self.state_shape = self.env.state_shape
        self.nA = self.env.nA

    def act(self, state, epsilon=0.01):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        pass

    def step(self, experience):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        pass

    def on_episode_begin(self, state):
        pass

    def on_episode_end(self, experiences):
        pass

    def on_env_solved(self):
        pass

    def save_model(self, filepath):
        raise NotImplementedError()

    def load_model(self, filepath):
        raise NotImplementedError()

    def get_model_ext(self):
        raise NotImplementedError()

