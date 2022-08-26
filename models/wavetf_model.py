from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Reshape, Conv2DTranspose, BatchNormalization, Activation, \
    AveragePooling2D, Concatenate, Lambda
from tensorflow.keras.models import Model
from wavetf import WaveTFFactory

from attacks.gaussian_noise_attack import gaussian_noise_function
from attacks.rotation_attack import rotation_function
from attacks.salt_pepper_attack import salt_pepper_function
from attacks.stupid_attack import stupid_function
from attacks.drop_out_attack import drop_out_function
from attacks.jpeg_attack import jpeg_function
from models.base_model import BaseModel


class WaveTFModel(BaseModel):

    def __init__(self, image_size: Tuple[int], watermark_size: Tuple[int], wavelet_type='haar'):
        # super(BaseModel, self).__init__(image_size=image_size, watermark_size=watermark_size)
        self.image_size = image_size
        self.watermark_size = watermark_size
        self.wavelet_type = wavelet_type
        self.preprocess_watermark_channels = [512, 128, 1]
        self.extraction_channels = [128, 256]
        self.preprocess_watermark_activation = 'relu'
        self.watermark_x_size = int(np.sqrt(self.watermark_size[0]))
        assert self.watermark_x_size == np.sqrt(self.watermark_size[0]), 'watermark cannot reshape square'

    def input_layers(self):
        image_input_layer = Input(self.image_size, name='image_input')
        watermark_input_layer = Input(self.watermark_size, name='watermark_input')
        attack_id_layer = Input((1,), name='attack_id_input', dtype='int32')
        return image_input_layer, watermark_input_layer, attack_id_layer

    def wavelet_transform(self, image_input_layer):
        wavelet_factory = WaveTFFactory().build(self.wavelet_type, dim=2)(image_input_layer)
        first_wavelet_image = wavelet_factory[:, :, :, 3:] / 2
        return first_wavelet_image, wavelet_factory

    @staticmethod
    def preprocess_image_network(image_input):
        image_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(image_input)
        return image_output

    def preprocess_watermark_network(self, watermark_input):
        square_watermark = Reshape(target_shape=(self.watermark_x_size, self.watermark_x_size, 1),
                                   input_shape=self.watermark_size, name='reshape_watermark')(watermark_input)
        for channels in self.preprocess_watermark_channels:
            square_watermark = Conv2DTranspose(filters=channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                square_watermark)
            square_watermark = BatchNormalization()(square_watermark)
            square_watermark = Activation(self.preprocess_watermark_activation)(square_watermark)
            square_watermark = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(square_watermark)
        return square_watermark

    @staticmethod
    def embedding_network(input_network):
        for i in range(3):
            input_network = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_network)
            input_network = BatchNormalization()(input_network)
            input_network = Activation('relu')(input_network)
        output_layer = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(input_network)
        output_layer = Activation('tanh')(output_layer)
        return output_layer

    def wavelet_inverse_transform(self, origin_image, watermarked_image):
        concatenate = Concatenate(axis=-1)([origin_image[:, :, :, :3], watermarked_image * 2])
        wavelet_inverse_layer = WaveTFFactory().build(self.wavelet_type, dim=2, inverse=True)(concatenate)
        return wavelet_inverse_layer

    @staticmethod
    def attack_simulator(image_layer, attack_id_layer):
        condition = Lambda(lambda x: tf.switch_case(
            x[0][0],
            branch_fns={
                0: lambda: stupid_function(x[1]),
                1: lambda: salt_pepper_function(x[1]),
                2: lambda: gaussian_noise_function(x[1]),
                3: lambda: jpeg_function(x[1]),
                4: lambda: drop_out_function(x[1])
            },
            default=lambda: stupid_function(x[1])
        ))((attack_id_layer[0], image_layer))
        return condition

    def extraction_network(self, watermarked_image):
        for channels in self.extraction_channels:
            watermarked_image = Conv2D(filters=channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                watermarked_image)
            watermarked_image = BatchNormalization()(watermarked_image)
            watermarked_image = Activation('relu')(watermarked_image)
        watermark = Conv2D(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same')(watermarked_image)
        watermark = Activation('sigmoid')(watermark)
        reshape_watermark = Reshape(target_shape=self.watermark_size,
                                    input_shape=(self.watermark_x_size, self.watermark_x_size, 1),
                                    name='output_watermark')(watermark)
        return reshape_watermark

    def get_model(self):
        image_input_layer, watermark_input_layer, attack_id_layer = self.input_layers()
        wavelet_image, whole_wavelet_image = self.wavelet_transform(image_input_layer)
        preprocessed_image = self.preprocess_image_network(wavelet_image)
        preprocessed_watermark = self.preprocess_watermark_network(watermark_input_layer)
        concatenate = Concatenate(axis=-1)([preprocessed_image, preprocessed_watermark])
        watermarked_image = self.embedding_network(concatenate)
        wavelet_inverse_watermarked_image = self.wavelet_inverse_transform(whole_wavelet_image, watermarked_image)
        wavelet_watermarked_image = Lambda(lambda x: x, name='embedded_image')(wavelet_inverse_watermarked_image)
        attack_layer = self.attack_simulator(wavelet_inverse_watermarked_image, attack_id_layer)
        wavelet_attack_image = WaveTFFactory().build(self.wavelet_type, dim=2)(attack_layer)
        first_channel_attack_image = wavelet_attack_image[:, :, :, 3:4] / 2
        extracted_watermark = self.extraction_network(first_channel_attack_image)
        return Model(
            inputs=[image_input_layer, watermark_input_layer, attack_id_layer],
            outputs=[wavelet_watermarked_image, extracted_watermark],
            name='embedding_network'
        )
