class Agent:

    def __init__(self, env):
        """ Initialize agent.

        Params
        ======
        - env: the environment, agent is supposed to interact with
        """
        self.env = env

        self.action_shape = self._extract_shape(env.action_space)
        self.state_shape = self._extract_shape(env.observation_space)

        action_type = type(env.action_space).__name__
        if action_type == 'Discrete':
            self.nA = env.action_space.n

        state_type = type(env.observation_space).__name__
        if state_type == 'Discrete':
            self.nS = env.observation_space.n

    def act(self, state, epsilon=0.01):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        raise NotImplementedError()

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        raise NotImplementedError()

    def _extract_shape(self, space):
        state_type = type(space).__name__
        if state_type == 'Discrete':
            return 1,
        return space.shape
