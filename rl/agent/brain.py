from abc import abstractmethod

import keras.backend as K
from keras import Input, Model
from keras.activations import relu
from keras.layers import Dense, Lambda
from keras.losses import mean_squared_error
from keras.optimizers import Adam

DEFAULT_LEARNING_RATE = 5e-3


class Brain:

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_model(self, state_shape, action_shape):
        raise NotImplementedError()


class DqnBrain(Brain):

    def __init__(self, optimizer=None) -> None:
        super().__init__()
        if optimizer is None:
            optimizer = Adam(DEFAULT_LEARNING_RATE)

        self.optimizer = optimizer

    def create_model(self, state_shape, action_shape):
        inputs = Input(shape=state_shape)

        x = inputs
        x = Dense(64, activation=relu)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(action_shape[0])(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss=mean_squared_error, optimizer=self.optimizer)
        return model


class DuelingDqnBrain(Brain):
    def __init__(self, optimizer=None) -> None:
        super().__init__()
        if optimizer is None:
            optimizer = Adam(DEFAULT_LEARNING_RATE)

        self.optimizer = optimizer

    def create_model(self, state_shape, action_shape):
        inputs = Input(shape=state_shape)
        x = inputs
        x = Dense(32, activation=relu)(x)

        x_value = Dense(32, activation=relu)(x)
        x_advantage = Dense(32, activation=relu)(x)

        value = Dense(1)(x_value)
        advantage = Dense(action_shape[0])(x_advantage)

        # V + (A - avg(A))
        q = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)))([value, advantage])
        
        model = Model(inputs=inputs, outputs=q)
        model.compile(loss=mean_squared_error, optimizer=self.optimizer)
        return model
