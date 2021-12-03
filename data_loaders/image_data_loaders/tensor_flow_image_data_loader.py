from pathlib import Path
from typing import List

import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader


class TensorFlowImageDataLoader(BaseDataLoader):
    def __init__(self, base_path: str, channels: List[int], convert_type=None):
        super(TensorFlowImageDataLoader, self).__init__()
        self.base_path = base_path
        self.channels = channels
        self.convert_type = convert_type

    def get_file_paths(self):
        file_paths = list(map(str, Path(self.base_path).glob('*.jpg')))
        tensor_file_paths = tf.data.from_tensor_slices((file_paths,))
        return tensor_file_paths

    def get_images(self, file_paths):
        images = tf.io.read_file(file_paths)
        decoded_images = tf.image.decode_jpeg(images, channels=3)
        selected_channels_images = decoded_images[:, :, self.channels]
        result_images = selected_channels_images
        if self.convert_type is not None:
            result_images = tf.image.convert_image_dtype(result_images, self.convert_type)
        return result_images

    def get_data_loader(self):
        file_paths = self.get_file_paths()
        image_loader = file_paths.map(self.get_images, num_parallel_calls=4)
        return image_loader
