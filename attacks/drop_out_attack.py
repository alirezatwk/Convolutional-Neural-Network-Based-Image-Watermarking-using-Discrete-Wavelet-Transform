import tensorflow as tf
from tensorflow.python import keras

from attacks.base_attack import BaseAttack


class DropOutAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(DropOutAttack, self).__init__()

    def drop_out(self, inputs):
        shp = keras.backend.shape(inputs)[1:]
        mask_select = tf.random.uniform(shape=shp ,maxval=1,dtype=tf.float32,seed=None)
        mask_select = mask_select > 0.3
        mask_noise = tf.cast(mask_select, tf.float32)
        out = inputs * mask_noise
        return out

    def call(self, inputs):
        outputs = self.drop_out(inputs)
        return outputs


def drop_out_function(x):
    return DropOutAttack()(x)
