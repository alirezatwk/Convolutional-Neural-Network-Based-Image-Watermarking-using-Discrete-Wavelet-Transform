import tensorflow as tf
from configs import *
from wavetf._haar_conv import HaarWaveLayer2D, InvHaarWaveLayer2D
from data_loaders.merged_data_loader import MergedDataLoader
from tensorflow.python.keras.models import load_model
from keras.callbacks import ModelCheckpoint

initial_epoch = 2
model_name = 'epochs:001-embedded_image_loss:0.002901714-output_watermark_loss:0.173016176.hdf5'

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
losses = {
    'embedded_image': 'mse',
    'output_watermark': 'mae',
}
loss_weights = {
    'embedded_image': IMAGE_LOSS_WEIGHT,
    'output_watermark': WATERMARK_LOSS_WEIGHT,
}

train_dataset = MergedDataLoader(image_base_path=TRAIN_IMAGES_PATH, image_channels=[0], image_convert_type=tf.float32,
                                 watermark_size=WATERMARK_SIZE, attack_max_id=5,
                                 batch_size=BATCH_SIZE).get_data_loader()

model = load_model(MODEL_OUTPUT_PATH + model_name,
                   custom_objects={"HaarWaveLayer2D": HaarWaveLayer2D, 'InvHaarWaveLayer2D': InvHaarWaveLayer2D})
file_path = MODEL_OUTPUT_PATH + 'epochs:{epoch:03d}-embedded_image_loss:{' \
                                'embedded_image_loss:.9f}-output_watermark_loss:{output_watermark_loss:.9f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1)
callbacks_list = [checkpoint]

model.fit(train_dataset, epochs=EPOCHS, callbacks=[callbacks_list], batch_size=BATCH_SIZE, initial_epoch=initial_epoch)
