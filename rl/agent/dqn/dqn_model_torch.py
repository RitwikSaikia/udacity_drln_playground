from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.backend_torch import _device
from .dqn_model import _AbstractDqnModel


class _TorchDqnModel(_AbstractDqnModel):
    _vars_initialized = False

    def __init__(self, input_shape, output_shape, optimizer=None) -> None:
        super().__init__(input_shape, output_shape)
        if optimizer is None:
            optimizer = torch.optim.Adam

        self.optimizer = optimizer
        self.loss_fn = F.mse_loss
        self._create(self.input_shape, self.output_shape)
        self.optimizer = optimizer(self._model.parameters(), lr=self.lr)

    def predict(self, X):
        X = torch.from_numpy(X).float()
        self._model.eval()
        with torch.no_grad():
            Y_pred = self._model(X)
        self._model.train()
        return Y_pred.numpy()

    def train(self, X, Y):
        X = torch.from_numpy(X).float()
        Y_pred = self._model(X)

        Y = torch.from_numpy(Y).float()
        loss = self.loss_fn(Y_pred, Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def get_weights(self):
        return [param.data for param in self._model.parameters()]

    def set_weights(self, weights):
        for i, param in enumerate(self._model.parameters()):
            weight = weights[i]
            param.data.copy_(weight)

    def _create(self, input_shape, output_shape):
        self._model = self._model_fn(input_shape, output_shape).to(_device())

    @abstractmethod
    def _model_fn(self, input_shape, output_shape):
        raise NotImplementedError()


class DqnModel(_TorchDqnModel):

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape[0], output_shape[0])

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape, fc1_units=64, fc2_units=64):
            super().__init__()
            self.fc1 = nn.Linear(input_shape, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.action = nn.Linear(fc2_units, output_shape)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.action(x)


class DuelingDqnModel(_TorchDqnModel):

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape[0], output_shape[0])

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape, fc1_units=32, fc2_units=32):
            super().__init__()

            self.state_fc = nn.Linear(input_shape, fc1_units)

            self.value_fc1 = nn.Linear(fc1_units, fc2_units)
            self.value_fc2 = nn.Linear(fc2_units, 1)

            self.advantage_fc1 = nn.Linear(fc1_units, fc2_units)
            self.advantage_fc2 = nn.Linear(fc2_units, output_shape)

        def forward(self, state):
            state = F.relu(self.state_fc(state))

            value = F.relu(self.value_fc1(state))
            value = self.value_fc2(value)

            advantage = F.relu(self.advantage_fc1(state))
            advantage = self.advantage_fc2(advantage)

            return value + (advantage - advantage.mean())


class DqnConvModel(_TorchDqnModel):

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape, output_shape[0])

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape):
            super().__init__()
            self.input_shape = input_shape

            self.block1_conv1 = nn.Conv2d(input_shape[2], 32, 8, stride=4)
            self.block1_bn1 = nn.BatchNorm2d(32)

            self.block2_conv1 = nn.Conv2d(32, 64, 4, stride=2)
            self.block2_bn1 = nn.BatchNorm2d(64)

            self.block3_conv1 = nn.Conv2d(64, 128, 4, stride=2)
            self.block3_bn1 = nn.BatchNorm2d(128)

            self.fc1 = nn.Linear(64 * 6 * 3, 256)
            self.action = nn.Linear(256, output_shape)

        def forward(self, x):
            shape = (-1,) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
            x = x.reshape(*shape)
            x = F.relu(self.block1_bn1(self.block1_conv1(x)))
            x = F.relu(self.block2_bn1(self.block2_conv1(x)))
            x = F.relu(self.block3_bn1(self.block3_conv1(x)))

            x = x.view(x.size(0), -1)  # Flatten

            x = F.relu(self.fc1(x))
            return self.action(x)
