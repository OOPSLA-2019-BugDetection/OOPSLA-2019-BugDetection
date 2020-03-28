from keras import backend as K
import tensorflow as tf
from keras.layers import Layer
from keras.layers.core import Lambda


class adding_weight(Layer):

    def __init__(self, output_len, output_dim, **kwargs):
        self.output_len = output_len
        self.output_dim = output_dim
        super(adding_weight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(adding_weight, self).build(input_shape)

    def call(self, x):
        temp = Lambda(lambda y: K.dot(y, self.kernel))(x[1])
        temp = Lambda(lambda y: tf.expand_dims(y, 1))(temp)
        temp = Lambda(lambda y: tf.tile(y, multiples=[1, self.output_len, 1]))(temp)
        return Lambda(lambda y: tf.multiply(x[0], y))(temp)

    def compute_output_shape(self, input_shape):
        return (None, self.output_len, self.output_dim)