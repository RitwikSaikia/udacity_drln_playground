from abc import abstractmethod

import tensorflow as tf
from tensorflow import layers as L
from tensorflow import nn as N
import numpy as np

from .dqn_model import _AbstractDqnModel
from ...backend_tf import _sess_config


class _TensorflowDqnModel(_AbstractDqnModel):
    _vars_initialized = False

    def __init__(self, input_shape, output_shape, optimizer=None) -> None:
        super().__init__(input_shape, output_shape)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.lr)

        self.optimizer = optimizer
        self.scope_name = "%s" % (self.__class__.__name__)
        self._graph = tf.Graph()
        with self._graph.as_default() as graph:
            with graph.name_scope(self.scope_name):
                self._create(self.input_shape, self.output_shape)
        tf.reset_default_graph()

        self._sess = tf.Session(config=_sess_config(), graph=self._graph)

    def __del__(self):
        self._sess.close()

    def predict(self, X):
        self._initialize_vars()
        X = np.reshape(X, (-1,) + self.input_shape)
        return self._sess.run(self._Y_pred, feed_dict={self._X: X})

    def train(self, X, Y):
        self._initialize_vars()
        result = self._sess.run([self._loss_op, self._train_op],
                                feed_dict={self._X: X, self._Y: Y})
        return result[0]

    def get_weights(self):
        self._initialize_vars()

        vars = self._get_weight_tensors()
        return self._sess.run(vars)

    def set_weights(self, weights):
        self._initialize_vars()

        feed_dict = {}
        assign_ops = []
        for i, (assign_op, assign_value) in enumerate(self._weight_assign_ops):
            weight = weights[i]
            assign_ops.append(assign_op)
            feed_dict[assign_value] = weight
        self._sess.run(assign_ops, feed_dict=feed_dict)

    def save_model(self, filename):
        saver = tf.train.Saver(self._weights, sharded=False)
        saver.save(self._sess, filename)

    def load_model(self, filename):
        self._initialize_vars()
        saver = tf.train.Saver(self._weights, sharded=False)
        saver.restore(self._sess, filename)

    def _get_weight_tensors(self):
        names = sorted(self._weights.keys())
        vars = [self._weights[n] for n in names]
        return vars

    def _initialize_vars(self):
        if self._vars_initialized:
            return
        uninitialized_vars = self._all_vars.values()
        uninitialized_vars = [v.initializer for v in uninitialized_vars]
        self._sess.run(uninitialized_vars)

        self._vars_initialized = True

    def _create(self, input_shape, output_shape):
        X, Y, Y_pred = self._model_fn(input_shape, output_shape)

        loss_op = tf.reduce_mean((Y - Y_pred) ** 2)
        train_op = self.optimizer.minimize(loss_op)

        weights = self._vars_to_dict(tf.trainable_variables())
        all_vars = self._vars_to_dict(tf.global_variables())

        self._X = X
        self._Y = Y
        self._Y_pred = Y_pred
        self._loss_op = loss_op
        self._train_op = train_op
        self._weights = weights
        self._all_vars = all_vars

        self._weight_assign_ops = self._create_weight_assign_ops()

    def _vars_to_dict(self, vars):
        return dict([(v.name, v) for v in vars])

    def _create_weight_assign_ops(self):
        assign_ops = []

        vars = self._get_weight_tensors()
        for i, var in enumerate(vars):
            assign_value = tf.placeholder(var.dtype, shape=var.shape)
            assign_op = tf.assign(var, assign_value)

            assign_ops.append((assign_op, assign_value))

        return assign_ops

    @abstractmethod
    def _model_fn(self, input_shape, output_shape):
        raise NotImplementedError()


class DqnModel(_TensorflowDqnModel):

    def _model_fn(self, input_shape, output_shape):
        X = tf.placeholder(tf.float32, (None,) + input_shape, name="state")
        Y = tf.placeholder(tf.float32, (None,) + output_shape, name="action_true")
        nn = X
        nn = L.dense(nn, 64, activation=N.relu, name="fc1")
        nn = L.dense(nn, 64, activation=N.relu, name="fc2")
        nn = L.dense(nn, output_shape[0], name="action")
        Y_pred = nn

        return X, Y, Y_pred


class DuelingDqnModel(_TensorflowDqnModel):

    def _model_fn(self, input_shape, output_shape):
        X = tf.placeholder(tf.float32, (None,) + input_shape, name="state")
        Y = tf.placeholder(tf.float32, (None,) + output_shape, name="action_true")

        nn = X
        nn = L.dense(nn, 32, activation=N.relu, name="fc1")

        value = L.dense(nn, 32, activation=N.relu, name="value_fc2")
        value = L.dense(value, 1, name="value")

        advantage = L.dense(nn, 32, activation=N.relu, name="advantage_fc2")
        advantage = L.dense(advantage, output_shape[0], name="advantage")

        q = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name="action")

        Y_pred = q

        return X, Y, Y_pred


class DqnConvModel(_TensorflowDqnModel):

    def _model_fn(self, input_shape, output_shape):
        X = tf.placeholder(tf.float32, (None,) + input_shape, name="state")
        Y = tf.placeholder(tf.float32, (None,) + output_shape, name="action_true")

        nn = X

        nn = L.conv2d(nn, 32, 8, strides=4, name="block1/conv1")
        nn = L.batch_normalization(nn, name="block1/bn")
        nn = N.relu(nn, name="block1/relu")

        nn = L.conv2d(nn, 64, 4, strides=2, name="block2/conv1")
        nn = L.batch_normalization(nn, name="block2/bn")
        nn = N.relu(nn, name="block2/relu")

        nn = L.conv2d(nn, 128, 4, strides=2, name="block3/conv1")
        nn = L.batch_normalization(nn, name="block3/bn")
        nn = N.relu(nn, name="block3/relu")

        nn = L.flatten(nn, name="flatten")
        nn = L.dense(nn, 256, activation=N.relu, name="fc1")
        nn = L.dense(nn, 256, activation=N.relu, name="fc2")
        nn = L.dense(nn, output_shape[0], name="action")

        Y_pred = nn

        return X, Y, Y_pred


class DuelingDqnConvModel(_TensorflowDqnModel):

    def _model_fn(self, input_shape, output_shape):
        X = tf.placeholder(tf.float32, (None,) + input_shape, name="state")
        Y = tf.placeholder(tf.float32, (None,) + output_shape, name="action_true")

        nn = X

        nn = L.conv2d(nn, 32, 8, strides=4, name="block1/conv1")
        nn = L.batch_normalization(nn, name="block1/bn")
        nn = N.relu(nn, name="block1/relu")

        nn = L.conv2d(nn, 64, 4, strides=2, name="block2/conv1")
        nn = L.batch_normalization(nn, name="block2/bn")
        nn = N.relu(nn, name="block2/relu")

        nn = L.conv2d(nn, 128, 4, strides=2, name="block3/conv1")
        nn = L.batch_normalization(nn, name="block3/bn")
        nn = N.relu(nn, name="block3/relu")

        nn = L.flatten(nn, name="flatten")
        nn = L.dense(nn, 32, activation=N.relu, name="fc1")

        value = L.dense(nn, 32, activation=N.relu, name="value_fc2")
        value = L.dense(value, 1, name="value")

        advantage = L.dense(nn, 32, activation=N.relu, name="advantage_fc2")
        advantage = L.dense(advantage, output_shape[0], name="advantage")

        q = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name="action")

        Y_pred = q

        return X, Y, Y_pred
