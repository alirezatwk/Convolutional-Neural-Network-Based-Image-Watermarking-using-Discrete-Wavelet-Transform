import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.configs import SEED


class TensorFlowWatermarkDataLoader(BaseDataLoader):
    def __init__(self, watermark_size, seed=SEED):
        super(TensorFlowWatermarkDataLoader, self).__init__()
        self.watermark_size = watermark_size
        self.seed = seed

    def watermark_generator(self):
        generator = tf.random.Generator.from_seed(self.seed)
        while True:
            yield tf.round(
                generator.uniform(self.watermark_size, 0, 1, dtype=tf.dtypes.float32)
            )

    def get_data_loader(self):
        watermark_loader = tf.data.Dataset.from_generator(
            self.watermark_generator,
            output_types=tf.float32,
            output_shapes=self.watermark_size
        )
        return watermark_loader
