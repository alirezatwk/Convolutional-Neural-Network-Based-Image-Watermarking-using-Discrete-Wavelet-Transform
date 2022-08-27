import tensorflow as tf
from tensorflow.keras.layers import Lambda

from attacks.base_attack import BaseAttack


class JPEGAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(JPEGAttack, self).__init__()

    def jpeg(self, inputs):
        images = tf.convert_to_tensor(inputs)
        batch = images.shape[0]
        outputs = []
        for i in range(batch):
            outputs.append(tf.image.adjust_jpeg_quality(images[i], 50))
        return tf.stack(outputs, axis=0)

    def call(self, inputs):
        outputs = Lambda(lambda x: self.jpeg(x), output_shape=(10, 256, 256, 1))(inputs)
        return outputs


def jpeg_function(x):
    return JPEGAttack()(x)
