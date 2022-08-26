from tensorflow.python import keras
import numpy as np
from attacks.base_attack import BaseAttack


class JPEGAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(JPEGAttack, self).__init__()
        self.q_mat = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]],
            dtype='float32'
        )
        self.quality = 50
        if self.quality < 50:
            self.s = 5000 / self.quality
        else:
            self.s = 200 - 2 * self.quality
        self.q_mat = np.floor((self.s * self.q_mat + 50.0) / 100.0)
        self.q_mat = np.reshape(self.q_mat, (64, 1))
        self.q_mat = np.repeat(self.q_mat[np.newaxis, ...])


    def jpeg(self, inputs):
        return inputs

    def call(self, inputs):
        outputs = self.jpeg(inputs)
        return outputs


def jpeg_function(x):
    return JPEGAttack()(x)
