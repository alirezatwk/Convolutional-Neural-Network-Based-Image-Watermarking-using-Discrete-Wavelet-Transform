import tensorflow as tf
from data_loaders.base_data_loader import BaseDataLoader
from typing import List
from data


class TensorFlowWatermarkDataLoader(BaseDataLoader):
    def __init__(self, watermark_size, seed=1234):
        super(TensorFlowWatermarkDataLoader, self).__init__()
        self.watermark_size = watermark_size
        self.seed = seed

    def watermark_generator(self):
        generator = tf.random.Generator.from_seed(self.seed)


    def get_data_loader(self):
        generator = tf.random.Generator.from_seed(self.seed)
        while True:
