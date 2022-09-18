from pathlib import Path
from typing import List

import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.configs import IMAGE_FORMATS


class ImageDataLoader(BaseDataLoader):
    def __init__(self, base_path: str, channels: List[int], convert_type=None, images_format=IMAGE_FORMATS):
        super(ImageDataLoader, self).__init__()
        self.base_path = base_path
        self.channels = channels
        self.convert_type = convert_type
        self.images_format = images_format

    def get_file_paths(self):
        file_paths = list(map(str, Path(self.base_path).glob(f'*.{self.images_format}')))[:39210] # TODO: Change this number as config
        labels = [0] * len(file_paths)
        tensor_file_paths = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        return tensor_file_paths

    def get_images(self, file_path, label):
        images = tf.io.read_file(file_path)
        decoded_images = tf.image.decode_jpeg(images, channels=3)
        selected_channels_images = decoded_images[:, :, 0]
        result_images = tf.reshape(selected_channels_images, [256, 256, 1])
        if self.convert_type is not None:
            result_images = tf.image.convert_image_dtype(result_images, self.convert_type)
        return result_images

    def get_data_loader(self):
        file_paths = self.get_file_paths()
        image_loader = file_paths.map(self.get_images, num_parallel_calls=4)
        return image_loader
