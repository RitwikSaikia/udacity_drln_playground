import numpy as np
import tensorflow as tf

from rl.agent.dqn.dqn_model import DqnModel

input_shape = (100,)
output_shape = (4,)

model = DqnModel(input_shape, output_shape)
model1 = DqnModel(input_shape, output_shape)

X = np.expand_dims(np.random.randn(*input_shape), axis=0)
Y = np.expand_dims(np.random.randn(*output_shape), axis=0)

with tf.Session() as sess:
    vars = tf.trainable_variables()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        cost = model.train(X, Y)

    Y_pred = model.predict(X)

print("Y      = ", Y)
print("Y_pred = ", Y_pred)
