from tensorflow.python import keras


class BaseAttack(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BaseAttack, self).__init__()

    def call(self, inputs):
        pass
