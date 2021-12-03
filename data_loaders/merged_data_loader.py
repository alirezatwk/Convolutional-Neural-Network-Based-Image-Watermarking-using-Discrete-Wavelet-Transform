from typing import List, Tuple

import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.configs import PREFETCH
from data_loaders.image_data_loaders.tensor_flow_image_data_loader import TensorFlowImageDataLoader
from data_loaders.watermark_data_loaders.tensor_flow_watermark_data_loader import TensorFlowWatermarkDataLoader


class MergedDataLoader(BaseDataLoader):
    def __init__(self, image_base_path: str, image_channels: List[int], image_convert_type, watermark_size: Tuple[int],
                 batch_size: int, prefetch=PREFETCH):
        super(MergedDataLoader, self).__init__()
        self.image_data_loader = TensorFlowImageDataLoader(base_path=image_base_path, channels=image_channels,
                                                           convert_type=image_convert_type)
        self.watermark_data_loader = TensorFlowWatermarkDataLoader(watermark_size=watermark_size)
        self.batch_size = batch_size
        self.prefetch = prefetch

    def get_data_loader(self):
        merged_data_loader = tf.data.Dataset.zip((self.image_data_loader, self.watermark_data_loader))
        merged_data_loader = merged_data_loader.batch(self.batch_size)
        merged_data_loader = merged_data_loader.prefetch(self.prefetch)
        return merged_data_loader
