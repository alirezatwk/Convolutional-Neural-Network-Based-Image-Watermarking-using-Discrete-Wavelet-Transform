import tensorflow as tf
from Model import FinalModel
from StringGenerator import stringGenerator
from configs import *
# Must implement loss function



model = FinalModel(IMAGE_SIZE, WATERMARK_SIZE)

losses = {
    "embeddedImage": 'mse',
    "outputWatermark": 'mae',
}
lossWeights = {
    "embeddedImage": 1.0,
    "outputWatermark": 1.0,
}
# MAE neveshte to maghale
# yesssss
# vali nemidonam chejori MAE ro benevisam
# https://www.tensorflow.org/api_docs/python/tf/losses
# in ja fekr konam hamashon hastan
#  tf.keras.losses.meansquarederror()
# aaaa
# chert goftam hale. nagooooo.
# nemidonam. daram charand migam shayadaa
# mae ina motmaen nistam bara 2D tarif beshe aslan!
# nemidonam hala bezanim hamin shekli? :)))
# are baba

# yes yes
XTrain, yTrain = getDataset(TRAINING_SIZE)
XTest, yTest = getDataset(TEST_SIZE)

model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
model.fit(XTrain, yTrain, epochs=EPOCHS)

model.evaluate(XTest, yTest)


