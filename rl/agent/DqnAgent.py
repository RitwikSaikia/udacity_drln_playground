import random

import numpy as np
from keras.utils import to_categorical

from rl.agent import Agent
from rl.util import ReplayBuffer, PrioritizedReplayBuffer


class DqnAgent(Agent):

    def __init__(self, env,
                 brain,
                 gamma=0.99,
                 tau=1e-3,
                 batch_size=64,
                 buffer_size=int(1e5),
                 update_every=4,
                 use_double_dqn=True,
                 use_prioritized_experience_replay=True):
        super().__init__(env)

        self.gamma = gamma
        self.prev_episode = 1
        self.epsilon = 1.0
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.t_step = 0
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        if self.use_prioritized_experience_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)

        self.qnetwork_local = brain.create_model(self.state_shape, self.action_shape)
        self.qnetwork_target = brain.create_model(self.state_shape, self.action_shape)

    def act(self, state, epsilon=0.01):
        if random.random() < epsilon:
            return self.env.sample_action()
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

        (indexes, weights_is), experiences = self.memory.sample(self.batch_size)
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
        Qs_local = self.qnetwork_local.predict(states)
        Qs_expected = Qs_local.copy()
        Qs_expected = Qs_expected * (1 - actions_one_hot) + actions_one_hot * (
                rewards + self.gamma * Qsa_next * (1 - dones))

        # TODO: use weights_is

        self.qnetwork_local.fit(states, Qs_expected, epochs=1, verbose=0)

        self.soft_update()

        if self.use_prioritized_experience_replay:
            delta_priority = np.squeeze(np.sum(np.abs(Qs_expected - Qs_local), axis=1))
            self.memory.update(indexes, delta_priority)

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
