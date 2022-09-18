import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader


class AttackIdDataLoader(BaseDataLoader):
    def __init__(self, max_value, min_value):
        super(AttackIdDataLoader, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def random_generator(self):
        while True:
            yield tf.random.uniform((1,), minval=self.min_value, maxval=self.max_value, dtype=tf.dtypes.int32)

    def get_data_loader(self):
        attack_id_loader = tf.data.Dataset.from_generator(
            self.random_generator,
            output_types=tf.int32,
            output_shapes=(1,)
        )
        return attack_id_loader
