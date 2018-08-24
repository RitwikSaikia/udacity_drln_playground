import random
from collections import deque, namedtuple

import numpy as np
from keras.utils import to_categorical

from rl.agent import Agent


class DqnAgent(Agent):

    def __init__(self, env,
                 create_model,
                 gamma=0.99,
                 learning_rate=5e-4,
                 tau=1e-3,
                 batch_size=64,
                 buffer_size=int(1e5),
                 update_every=4,
                 use_double_dqn=True):
        super().__init__(env)

        self.memory = ReplayBuffer(self.nA, buffer_size, batch_size)

        self.gamma = gamma
        self.prev_episode = 1
        self.epsilon = 1.0
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.t_step = 0
        self.use_double_dqn = use_double_dqn

        self.qnetwork_local = create_model(self.nA, self.state_shape, self.learning_rate)
        self.qnetwork_target = create_model(self.nA, self.state_shape, self.learning_rate)

    def act(self, state, epsilon=0.01):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        state = np.expand_dims(np.asarray(state), axis=0)
        return np.argmax(self.qnetwork_local.predict(state)[0])

    def step(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)
        self.learn()

    def learn(self):
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step != 0:
            return

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) <= self.batch_size:
            return

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        if self.use_double_dqn:
            Qs_local_next = self.qnetwork_local.predict(next_states)
            Qs_target_next = self.qnetwork_target.predict(next_states)
            local_next_actions = np.argmax(Qs_local_next, axis=1)

            next_local_actions_one_hot = to_categorical(local_next_actions, self.nA)
            Qsa_next = np.expand_dims(np.sum(Qs_target_next * next_local_actions_one_hot, axis=1), axis=1)
        else:
            Qsa_next = np.expand_dims(np.max(self.qnetwork_target.predict(next_states), axis=1), axis=1)

        actions_one_hot = to_categorical(np.squeeze(actions), self.nA)
        Qs_expected = self.qnetwork_local.predict(states)
        Qs_expected = Qs_expected * (1 - actions_one_hot) + actions_one_hot * (
                    rewards + self.gamma * Qsa_next * (1 - dones))

        self.qnetwork_local.fit(states, Qs_expected, epochs=1, verbose=0)

        self.soft_update()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        local_weights = self.qnetwork_local.get_weights()
        target_weights = self.qnetwork_target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * local_weights[i] + (1 - self.tau) * target_weights[i]
        self.qnetwork_target.set_weights(target_weights)

    def save_model(self, filepath):
        self.qnetwork_target.save(filepath)

    def load_model(self, filepath):
        self.qnetwork_target.load_weights(filepath)
        self.qnetwork_local.set_weights(self.qnetwork_target.get_weights())


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        :param action_size: dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def remember(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
