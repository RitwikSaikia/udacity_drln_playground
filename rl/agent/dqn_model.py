from abc import abstractmethod

import keras.backend as K
from keras import Input, Model
from keras.activations import relu
from keras.layers import Dense, Lambda, Conv2D, BatchNormalization, Activation, Flatten
from keras.losses import mean_squared_error
from keras.optimizers import Adam

DEFAULT_LEARNING_RATE = 5e-3


class AbstractDqnModel:

    def __init__(self, optimizer) -> None:
        super().__init__()

        if optimizer is None:
            optimizer = Adam(DEFAULT_LEARNING_RATE)

        self.optimizer = optimizer

    @abstractmethod
    def create_model(self, state_shape, action_shape):
        raise NotImplementedError()


class DqnModel(AbstractDqnModel):

    def __init__(self, optimizer=None) -> None:
        super().__init__(optimizer)

    def create_model(self, state_shape, action_shape):
        inputs = Input(shape=state_shape)

        x = inputs
        x = Dense(64, activation=relu)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(action_shape[0])(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss=mean_squared_error, optimizer=self.optimizer)
        return model


class DuelingDqnModel(AbstractDqnModel):
    def __init__(self, optimizer=None) -> None:
        super().__init__(optimizer)

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


class DqnConvModel(AbstractDqnModel):

    def __init__(self, optimizer=None) -> None:
        super().__init__(optimizer)

    def create_model(self, state_shape, action_shape):
        inputs = Input(shape=state_shape[1:])

        x = inputs
        x = Conv2D(32, 8, strides=4)(x)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        x = Conv2D(64, 4, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        x = Conv2D(128, 4, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)

        x = Flatten()(x)
        x = Dense(256, activation=relu)(x)
        x = Dense(256, activation=relu)(x)
        x = Dense(action_shape[0])(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss=mean_squared_error, optimizer=self.optimizer)
        return model
