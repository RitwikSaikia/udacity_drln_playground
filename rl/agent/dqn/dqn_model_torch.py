from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.backend_torch import _device
from .dqn_model import _AbstractDqnModel


class _TorchDqnModel(_AbstractDqnModel):

    def __init__(self, input_shape, output_shape, optimizer=None) -> None:
        super().__init__(input_shape, output_shape)
        if optimizer is None:
            def optimizer(model_params):
                return torch.optim.Adam(model_params,
                                        lr=self.lr)

        self._device = _device()
        self.optimizer = optimizer
        self.loss_fn = F.mse_loss
        self._create(self.input_shape, self.output_shape)
        self.optimizer = optimizer(self._model.parameters())

    def predict(self, states):
        states = torch.from_numpy(states).float().to(self._device)
        self._model.eval()
        with torch.no_grad():
            action_values = self._model(states)
        self._model.train()
        return action_values.cpu().numpy()

    def fit(self, states, actions, Qsa_expected):
        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).long().to(self._device)
        Qsa_expected = torch.from_numpy(Qsa_expected).float().to(self._device)

        # Get expected Q values from local model
        Qsa = self._model(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(Qsa_expected, Qsa)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_weights(self):
        return [param.data for param in self._model.parameters()]

    def set_weights(self, weights):
        for i, param in enumerate(self._model.parameters()):
            weight = weights[i]
            param.data.copy_(weight)

    def save_model(self, filename):
        torch.save(self._model.state_dict(), filename)

    def load_model(self, filename):
        self._model.load_state_dict(torch.load(filename))

    def _create(self, input_shape, output_shape):
        self._model = self._model_fn(input_shape, output_shape).to(self._device)

    @abstractmethod
    def _model_fn(self, input_shape, output_shape):
        raise NotImplementedError()


class DqnModel(_TorchDqnModel):

    def __init__(self, input_shape, output_shape, optimizer=None, fc_units=(64, 64,)) -> None:
        self.fc_units = fc_units
        super().__init__(input_shape, output_shape, optimizer)

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape[0], output_shape[0], fc_units=self.fc_units)

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape, fc_units):
            super().__init__()
            in_shape = input_shape
            self.fcs = nn.ModuleList()
            for f in fc_units:
                self.fcs.append(nn.Linear(in_shape, f))
                in_shape = f

            self.action = nn.Linear(in_shape, output_shape)

        def forward(self, x):
            for fc in self.fcs:
                x = F.relu(fc(x))
            return self.action(x)


class DuelingDqnModel(_TorchDqnModel):

    def __init__(self, input_shape, output_shape, optimizer=None, fc_units=(64, 32, 32)) -> None:
        self.fc_units = fc_units
        super().__init__(input_shape, output_shape, optimizer)

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape[0], output_shape[0], fc_units=self.fc_units)

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape, fc_units):
            super().__init__()
            in_shape = input_shape
            self.fcs = nn.ModuleList()
            for f in fc_units[:-1]:
                self.fcs.append(nn.Linear(in_shape, f))
                in_shape = f

            fc_units_last = fc_units[-1]

            self.value_fc1 = nn.Linear(in_shape, fc_units_last)
            self.value_fc2 = nn.Linear(fc_units_last, 1)

            self.advantage_fc1 = nn.Linear(in_shape, fc_units_last)
            self.advantage_fc2 = nn.Linear(fc_units_last, output_shape)

        def forward(self, x):
            for fc in self.fcs:
                x = F.relu(fc(x))

            state_fc = x

            value = F.relu(self.value_fc1(state_fc))
            value = self.value_fc2(value)

            advantage = F.relu(self.advantage_fc1(state_fc))
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
            self.block2_conv1 = nn.Conv2d(32, 64, 4, stride=2)
            self.block3_conv1 = nn.Conv2d(64, 64, 3, stride=1)

            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.action = nn.Linear(512, output_shape)

        def forward(self, x):
            shape = (-1,) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
            x = x.reshape(*shape)
            x = F.relu(self.block1_conv1(x))
            x = F.relu(self.block2_conv1(x))
            x = F.relu(self.block3_conv1(x))

            x = x.view(x.size(0), -1)  # Flatten

            x = F.relu(self.fc1(x))
            return self.action(x)


class DuelingDqnConvModel(_TorchDqnModel):

    def _model_fn(self, input_shape, output_shape):
        return self.NNModule(input_shape, output_shape[0])

    class NNModule(nn.Module):
        def __init__(self, input_shape, output_shape):
            super().__init__()
            self.input_shape = input_shape

            self.block1_conv1 = nn.Conv2d(input_shape[2], 32, 8, stride=4)
            self.block2_conv1 = nn.Conv2d(32, 64, 4, stride=2)
            self.block3_conv1 = nn.Conv2d(64, 64, 3, stride=1)

            self.state_fc = nn.Linear(64 * 7 * 7, 512)

            self.value_fc1 = nn.Linear(512, 64)
            self.value_fc2 = nn.Linear(64, 1)

            self.advantage_fc1 = nn.Linear(512, 64)
            self.advantage_fc2 = nn.Linear(64, output_shape)

        def forward(self, x):
            shape = (-1,) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
            x = x.reshape(*shape)
            x = F.relu(self.block1_conv1(x))
            x = F.relu(self.block2_conv1(x))
            x = F.relu(self.block3_conv1(x))

            x = x.view(x.size(0), -1)  # Flatten

            state = F.relu(self.state_fc(x))

            value = F.relu(self.value_fc1(state))
            value = self.value_fc2(value)

            advantage = F.relu(self.advantage_fc1(state))
            advantage = self.advantage_fc2(advantage)

            return value + (advantage - advantage.mean())
