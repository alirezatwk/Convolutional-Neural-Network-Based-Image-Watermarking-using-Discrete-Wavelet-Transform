import tensorflow_addons as tfa

from attacks.base_attack import BaseAttack


class RotationAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(RotationAttack, self).__init__()

    def rotation(self, inputs):
        # angles = np.random.randint(0, 91, (BATCH_SIZE,))
        angles = 90
        return tfa.image.rotate(
            images=inputs,
            angles=angles,
            fill_value=0.0,
        )

    def call(self, inputs):
        outputs = self.rotation(inputs)
        return outputs


def rotation_function(x):
    return RotationAttack()(x)
