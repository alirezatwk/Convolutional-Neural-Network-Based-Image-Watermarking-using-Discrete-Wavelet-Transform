import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Activation 
from tensorflow.python.keras.layers import BatchNormalization 
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Lambda 
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import AveragePooling2D 
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.models import Model

def FinalModel(imagesShape, watermarksShape):
    imageLayer = Input(imagesShape)
    watermarkLayer = Input(watermarksShape)

    x = hostImageNetwork(imageLayer)
    y = WMNetwork(watermarkLayer)

    con = Concatenate(axis=-1)([x, y])

    WMImage = embeddingNetwork(con)
    ext = extractionNetwork(WMImage)

    model = Model(inputs=[imageLayer, watermarkLayer], outputs=[WMImage, ext], name="embedingNetwork")
    # model.summary()
    return model

def hostImageNetwork(imageLayer):
    x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(imageLayer) 
    x = Conv2D(64, (3, 3), padding="same")(x)
    return x

def WMNetwork(watermarkLayer):
    # 4 layers : 1,2,3) CONV => RELU => POOL 4) CL => POOL
    y = Reshape((32, 32, 1), input_shape=(1024,))(watermarkLayer)
    
    # First layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(512, (3, 3), padding="same", strides=(1, 1))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(y)

    # Second layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(256, (3, 3), padding="same", strides=(1, 1))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(y)
    
    # Third layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(128, (3, 3), padding="same", strides=(1, 1))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(y)

    # Fourth layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(1, (3, 3), padding="same", strides=(1, 1))(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(y)
    return y

def embeddingNetwork(con):
    # 4 layers : 1,2,3) CONV => BN => RELU 4) CL => TANH

    # First layer
    con = Conv2D(64, (3, 3))(con)
    con = BatchNormalization()(con)
    con = Activation('relu')(con)

    # Second layer
    con = Conv2D(64, (3, 3))(con)
    con = BatchNormalization()(con)
    con = Activation('relu')(con)

    # Third layer
    con = Conv2D(64, (3, 3))(con)
    con = BatchNormalization()(con)
    con = Activation('relu')(con)
    
    # Fourth layer
    con = Conv2D(1, (3, 3))(con)
    WMImage = Activation('tanh', name="embeddedImage")(con)
    return WMImage

def extractionNetwork(WMImage):
    # 4 layers : 1,2,3) CONV => BN => ReLU 4) CL => tanh

    # First layer
    ext = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(WMImage)
    ext = BatchNormalization()(ext)
    ext = Activation('relu')(ext)

    # Second layer
    ext = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(ext)
    ext = BatchNormalization()(ext)
    ext = Activation('relu')(ext)

    # Third layer
    ext = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(ext)
    ext = BatchNormalization()(ext)
    ext = Activation('relu')(ext)
    
    # Fourth layer
    ext = Conv2D(1, (3, 3), strides=(2, 2), padding="same")(ext)
    ext = Activation('tanh', name="outputWatermark")(ext)
    return ext

# FinalModel((512, 512, 3), (32 * 32,))