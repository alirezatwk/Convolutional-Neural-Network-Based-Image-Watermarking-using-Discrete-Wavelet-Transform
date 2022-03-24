from attacks.base_attack import BaseAttack


class StupidAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(StupidAttack, self).__init__()

    def call(self, inputs):
        outputs = inputs
        return outputs


def stupid_function(x):
    return StupidAttack()(x)