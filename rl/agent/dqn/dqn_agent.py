import random

import numpy as np

from rl import get_backend
from ..agent import _AbstractAgent
from ...util import ReplayBuffer, PrioritizedReplayBuffer


class DqnAgent(_AbstractAgent):

    def __init__(self, env,
                 create_model_fn,
                 gamma=0.99,
                 tau=1e-3,
                 batch_size=64,
                 buffer_size=int(1e5),
                 update_every=4,
                 use_double_dqn=True,
                 use_prioritized_experience_replay=True,
                 use_importance_sampling=True):
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
        self.use_importance_sampling = use_importance_sampling

        if self.use_prioritized_experience_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)

        self.qnetwork_local = create_model_fn(self.state_shape, self.action_shape)
        self.qnetwork_target = create_model_fn(self.state_shape, self.action_shape)

    def act(self, state, epsilon=0.01):
        if random.random() < epsilon:
            return self.env.sample_action()
        state = np.reshape(np.asarray(state), self.state_shape)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        # Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)
        #
        # In comparison to Double Q-learning, the weights of the second network θ′
        # are replaced with the weights of the target network θ− for the evaluation
        # of the current greedy policy. The update to the target network stays unchanged
        # from DQN, and remains a periodic copy of the online network.
        net = self.qnetwork_target if self.use_double_dqn else self.qnetwork_local

        return np.argmax(net.predict(state)[0])

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

        update_params, experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        Qsa_next = np.expand_dims(np.max(self.qnetwork_target.predict(next_states), axis=1), axis=1)
        Qsa_expected = rewards + self.gamma * Qsa_next * (1 - dones)

        errors = None
        if self.use_prioritized_experience_replay:
            Qs_local = self.qnetwork_local.predict(states)
            action_indices = np.expand_dims(range(self.batch_size), axis=1) * self.nA + actions
            Qsa_local = np.take(Qs_local, action_indices)
            errors = np.sum(np.abs(Qsa_expected - Qsa_local), axis=1)
            update_params["errors"] = errors

        self.memory.update(**update_params)

        self.qnetwork_local.fit(states, actions, Qsa_expected)

        if self.use_prioritized_experience_replay and self.use_importance_sampling:
            # hyperparams
            eta = 0.1
            min_delta_is_threshold = 1e-12

            weights_is = update_params["weights_is"]

            # TODO: fix the importance sampling implementation
            # ∆  = ∆ + wj · δj · ∇Q(Sj−1,Aj−1)
            delta_is = np.sum(np.dot(weights_is, errors))
            if delta_is >= min_delta_is_threshold:
                local_weights = self.qnetwork_local.get_weights()
                local_weights = [delta_is * eta + w for w in local_weights]
                self.qnetwork_local.set_weights(local_weights)

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
        filepath = filepath + self._get_ext()
        self.qnetwork_target.save_model(filepath)
        return filepath

    def load_model(self, filepath):
        filepath = filepath + self._get_ext()
        self.qnetwork_target.load_model(filepath)
        self.qnetwork_local.load_model(filepath)
        return filepath

    def _get_ext(self):
        ext = ""
        backend = get_backend()
        if backend == 'tf':
            ext = '.tf.ckpt'
        elif backend == 'torch':
            ext = '.torch.pth'
        return ext
