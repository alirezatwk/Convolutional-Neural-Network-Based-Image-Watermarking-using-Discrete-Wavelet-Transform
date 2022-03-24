from tensorflow.python import keras

from attacks.base_attack import BaseAttack


class SaltPepperAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(SaltPepperAttack, self).__init__()

    def salt_pepper(self, inputs):
        shp = keras.backend.shape(inputs)[1:]
        mask_select = keras.backend.random_binomial(shape=shp, p=.1)
        mask_noise = keras.backend.random_binomial(shape=shp, p=0.5)  # salt and pepper have the same chance
        out = inputs * (1 - mask_select) + mask_noise * mask_select
        return out

    def call(self, inputs):
        outputs = self.salt_pepper(inputs)
        return outputs


def salt_pepper_function(x):
    return SaltPepperAttack()(x)
