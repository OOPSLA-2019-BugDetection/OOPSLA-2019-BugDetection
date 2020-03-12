from keras import backend as k
from keras.layers import Layer


class adding_weight(Layer):

    def __init__(self, **kwargs):
        super(adding_weight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                      shape=(input_shape[0], input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        super(adding_weight, self).build(input_shape)

    def call(self, x):
        return k.multiply(x, self.weight)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])