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

        X = np.expand_dims(np.random.randn(*input_shape), axis=0)
        Y = np.expand_dims(np.random.randn(*output_shape), axis=0)

        self.train_and_eval(X, Y, model)

    @parameterized.expand([
        [DqnConvModel]
    ])
    def test_conv_dqn_model(self, model):

        input_shape = (84, 84, 4,)
        output_shape = (4,)

        X = np.expand_dims(np.random.randn(*input_shape), axis=0)
        Y = np.expand_dims(np.random.randn(*output_shape), axis=0)

        self.train_and_eval(X, Y, model)

    def train_and_eval(self, X, Y, model):
        input_shape = X.shape[1:]
        output_shape = Y.shape[1:]

        local_model = model(input_shape, output_shape)
        target_model = model(input_shape, output_shape)
        empty_model = model(input_shape, output_shape)

        while True:
            loss = local_model.train(X, Y)
            if loss < 1e-5:
                break
        target_model.set_weights(local_model.get_weights())
        model_file = "/tmp/%s.%s.ckpt" % (self.__class__.__name__, model.__name__)
        if os.path.exists(model_file):
            os.remove(model_file)

        target_model.save_model(model_file)
        empty_model.load_model(model_file)

        Y_pred = empty_model.predict(X)
        loss = np.mean(np.square(Y_pred - Y))
        self.assertTrue(loss < 1e-3, "loss is very high = %f" % loss)


if __name__ == '__main__':
    unittest.main()
