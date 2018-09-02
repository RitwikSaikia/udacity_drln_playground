import os
import sys

sys.path.extend(("./", "../"))

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

        self.train_and_eval(input_shape, output_shape, model)

    @parameterized.expand([
        [DqnConvModel]
    ])
    def test_conv_dqn_model(self, model):

        input_shape = (84, 84, 4,)
        output_shape = (4,)

        self.train_and_eval(input_shape, output_shape, model)

    def train_and_eval(self, input_shape, output_shape, model):
        local_model = model(input_shape, output_shape)
        target_model = model(input_shape, output_shape)
        empty_model = model(input_shape, output_shape)

        batch_size = 16
        input_shape = (batch_size,) + input_shape
        output_shape = (batch_size,) + output_shape

        X = np.random.randn(*input_shape)
        actions = np.random.randint(0, output_shape[1] - 1, (batch_size, 1,))
        Qs_expected = np.random.randn(batch_size, 1)

        for i in range(int(1e2)):
            local_model.fit(X, actions, Qs_expected)

        Y_expected = local_model.predict(X)

        target_model.set_weights(local_model.get_weights())
        model_file = "/tmp/%s.%s.ckpt" % (self.__class__.__name__, model.__name__)
        if os.path.exists(model_file):
            os.remove(model_file)

        target_model.save_model(model_file)
        empty_model.load_model(model_file)

        Y_pred = empty_model.predict(X)
        loss = np.mean(np.square(Y_pred - Y_expected))
        self.assertTrue(loss == 0, "loss is very high = %f" % loss)


if __name__ == '__main__':
    unittest.main()
