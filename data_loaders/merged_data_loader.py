from typing import List, Tuple

import tensorflow as tf

from data_loaders.attack_id_data_loader.attack_id_data_loader import AttackIdDataLoader
from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.configs import PREFETCH
from data_loaders.image_data_loaders.image_data_loader import ImageDataLoader
from data_loaders.watermark_data_loaders.watermark_data_loader import WatermarkDataLoader


class MergedDataLoader(BaseDataLoader):
    def __init__(self, image_base_path: str, image_channels: List[int], image_convert_type, watermark_size: Tuple[int],
                 attack_min_id: int, attack_max_id: int, batch_size: int, prefetch=PREFETCH):
        super(MergedDataLoader, self).__init__()
        self.image_data_loader = ImageDataLoader(base_path=image_base_path, channels=image_channels,
                                                 convert_type=image_convert_type).get_data_loader()
        self.watermark_data_loader = WatermarkDataLoader(watermark_size=watermark_size).get_data_loader()
        self.attack_id_data_loader = AttackIdDataLoader(min_value=attack_min_id,
                                                        max_value=attack_max_id).get_data_loader()
        self.batch_size = batch_size
        self.prefetch = prefetch

    def get_data_loader(self):
        input_data_loader = tf.data.Dataset.zip((
            self.image_data_loader,
            self.watermark_data_loader,
            self.attack_id_data_loader,
        ))
        output_data_loader = tf.data.Dataset.zip((self.image_data_loader, self.watermark_data_loader))
        merged_data_loader = tf.data.Dataset.zip((input_data_loader, output_data_loader))
        merged_data_loader = merged_data_loader.batch(self.batch_size)
        merged_data_loader = merged_data_loader.prefetch(self.prefetch)
        return merged_data_loader
