import sys

sys.path.append("../")

import unittest
from rl import DuelingDqnModel, DqnModel, DqnConvModel
import numpy as np
from parameterized import parameterized


class TestDqnModel(unittest.TestCase):

    def setUp(self):
        pass

    @parameterized.expand([
        [DqnModel],
        [DuelingDqnModel]
    ])
    def test_dqn_model(self, model):

        input_shape = (100,)
        output_shape = (4,)

        local_model = model(input_shape, output_shape)
        target_model = model(input_shape, output_shape)

        X = np.expand_dims(np.random.randn(*input_shape), axis=0)
        Y = np.expand_dims(np.random.randn(*output_shape), axis=0)

        while True:
            loss = local_model.train(X, Y)
            if loss < 1e-5:
                break

        target_model.set_weights(local_model.get_weights())

        Y_pred = target_model.predict(X)

        loss = np.mean(np.square(Y_pred - Y))
        self.assertTrue(loss < 1e-3)

    @parameterized.expand([
        [DqnConvModel]
    ])
    def test_conv_dqn_model(self, model):

        input_shape = (84, 84, 4,)
        output_shape = (4,)

        local_model = model(input_shape, output_shape)
        target_model = model(input_shape, output_shape)

        X = np.expand_dims(np.random.randn(*input_shape), axis=0)
        Y = np.expand_dims(np.random.randn(*output_shape), axis=0)

        while True:
            loss = local_model.train(X, Y)
            if loss < 1e-5:
                break

        target_model.set_weights(local_model.get_weights())

        Y_pred = target_model.predict(X)

        loss = np.mean(np.square(Y_pred - Y))
        self.assertTrue(loss < 1e-3)


if __name__ == '__main__':
    unittest.main()
