import random
from abc import abstractmethod
from collections import deque

import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from unityagents import UnityEnvironment

from rl.util import to_one_hot


class Env:
    action_shape = None
    state_shape = None
    nA = None

    def __init__(self, seed, headless=False, train_mode=False) -> None:
        super().__init__()
        self._seed = seed
        self._headless = headless
        self._train_mode = train_mode

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        return NotImplementedError()

    @abstractmethod
    def sample_action(self):
        return random.choice(range(self.action_shape[0]))


class GymEnv(Env):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self._env = gym.make(name)
        self._env.seed(self._seed)
        self.action_shape = (self._env.action_space.n,)
        self.state_shape = self._extract_shape(self._env.observation_space)

        self.nA = self._env.action_space.n

    def reset(self):
        state = self._env.reset()
        state = self._process_state(state)
        return state

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        next_state = self._process_state(next_state)
        return next_state, reward, done, _

    def render(self, **kwargs):
        if not self._headless:
            self._env.render(**kwargs)

    def close(self):
        self._env.close()

    def _extract_shape(self, space):
        state_type = type(space).__name__
        if state_type == 'Discrete':
            self.nS = space.n
            return self.nS,
        return space.shape

    def _process_state(self, state):
        if hasattr(self, 'nS'):
            return to_one_hot(state, self.nS)
        return state


class UnityEnv(Env):
    allowed_modes = ['vector', 'visual']

    def __init__(self, filename: str, mode='vector',
                 frame_size=(84, 84), use_grayscale=True, n_frames=4,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if mode not in self.allowed_modes:
            raise Exception("Allowed modes : %s" % self.allowed_modes)

        if "headless" in kwargs:
            del kwargs["headless"]

        if "train_mode" in kwargs:
            del kwargs["train_mode"]

        self.mode = mode
        self.env = UnityEnvironment(filename, no_graphics=self._headless, **kwargs)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=self._train_mode)[self.brain_name]

        self.nA = brain.vector_action_space_size
        self.action_shape = (self.nA,)

        if mode == 'vector':
            self.nS = len(env_info.vector_observations[0])
            self.state_shape = (self.nS,)
        elif mode == 'visual':
            self.frame_size = tuple(frame_size)
            self.use_grayscale = use_grayscale
            self.n_frames = n_frames
            self.frame_buffer = deque(maxlen=self.n_frames)
            num_channels = 1
            if not use_grayscale:
                num_channels = 3
            self.state_shape = self.frame_size + (num_channels * n_frames,)

    def reset(self):
        if self.mode == 'visual':
            self.frame_buffer.clear()
        env_info = self.env.reset(train_mode=self._train_mode)[self.brain_name]
        return self._to_state(env_info)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = self._to_state(env_info)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        return next_state, reward, done, env_info

    def render(self, **kwargs):
        pass

    def close(self):
        pass

    def _process_frame(self, frame):
        frame = np.squeeze(frame, axis=0)
        frame = resize(frame, self.frame_size, mode='constant', anti_aliasing=True)
        if self.use_grayscale:
            frame = np.expand_dims(rgb2gray(frame), axis=2)
        return frame

    def _to_state(self, env_info):
        if self.mode == 'vector':
            return env_info.vector_observations[0]
        elif self.mode == 'visual':
            frame = self._process_frame(env_info.visual_observations[0])
            if len(self.frame_buffer) == 0:
                for i in range(self.n_frames):
                    self.frame_buffer.append(frame)
            else:
                self.frame_buffer.append(frame)

            result = np.reshape(self.frame_buffer, self.state_shape)
            result = np.expand_dims(result, axis=0)
            return result
