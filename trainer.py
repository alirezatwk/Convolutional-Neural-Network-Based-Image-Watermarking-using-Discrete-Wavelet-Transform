import tensorflow as tf
from configs import *
from models.wavetf_model import WaveTFModel
from data_loaders.merged_data_loader import MergedDataLoader
from keras.callbacks import ModelCheckpoint

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
                                 watermark_size=WATERMARK_SIZE, batch_size=BATCH_SIZE)

model = WaveTFModel(image_size=IMAGE_SIZE, watermark_size=WATERMARK_SIZE).get_model()
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
file_path = MODEL_OUTPUT_PATH + 'epochs:{epoch03d}-embedded_image_loss:{embedded_image_loss:.9f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1)
callbacks_list = [checkpoint]

model.fit(train_dataset, epochs=EPOCHS, callbacks=[callbacks_list], batch_size=BATCH_SIZE)
