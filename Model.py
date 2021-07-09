# from IPython.core.display import Image
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda, Reshape
from tensorflow.python.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image

# from tensorflow.python.keras.utils import layer_utils
# from tensorflow.python.keras.utils.data_utils import get_file
# from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model

#from kt_utils import * # Has problem

# import keras.backend as K
# K.set_image_data_format('channels_last')

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# from tensorflow.python.ops.gen_array_ops import size


def FinalModel(imagesShape, watermarksShape):

    images = Input(imagesShape)
    watermarks = Input(watermarksShape)

    x = hostImageNetwork(images)
    y = WMNetwork(watermarks)

    # Concatenate
    con = Concatenate(axis=-1)([x, y])

    WMImage = embeddingNetwork(con)
    ext = extractionNetwork(WMImage)

    model = Model(inputs=[images, watermarks], outputs=[WMImage, ext], name="embedingNetwork")
    # model.summary()
    return model

def hostImageNetwork(images):
    x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(images) 
    x = Conv2D(64, (3, 3), padding="same")(x)
    return x

def WMNetwork(watermarks):
    # 4 layers : 1,2,3) CONV => RELU => POOL 4) CL => POOL
    y = Reshape((32, 32, 1), input_shape=(1024,))(watermarks)
    
    # First layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(512, (3, 3), padding="same", strides=(1, 1))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(y)

    # Second layer
    y = UpSampling2D(size=(2, 2))(y) # Use for covering stride = 0.5 of Conv2D
    y = Conv2D(256, (3, 3), padding="same", strides=(1, 1))(y) # 128 according to Figure 2 TODO: ask Zahra what to do with this number
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




    # # CONV => RELU => POOL
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = MaxPooling2D(pool_size=(3, 3))(x)
    # x = Dropout(0.25)(x)

    # # (CONV => RELU) * 2 => POOL
    # x = Conv2D(64, (3, 3), padding="same")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = Conv2D(64, (3, 3), padding="same")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)

    # # (CONV => RELU) * 2 => POOL
    # x = Conv2D(128, (3, 3), padding="same")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = Conv2D(128, (3, 3), padding="same")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)

    # # define a branch of output layers for the number of different
    # # clothing categories (i.e., shirts, jeans, dresses, etc.)
    # x = Flatten()(x)
    # x = Dense(256)(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(numCategories)(x)
    # x = Activation(finalAct, name="category_output")(x)
    # return the category prediction sub-network
 #   return x


def networkModel(input_shape):
    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


class model(Model):
    
    def __init__(self):
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[1])
        self.l2=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(1,kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        
    # Running the model
    def run(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return boom2
      
    #Custom loss fucntion
    def get_loss(self,X,Y):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return tf.math.square(boom2-Y)
      
    # get gradients
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]])
        return g
      
    # perform gradient descent
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]]))



# define two sets of inputs
# inputA = Input(shape=(32,))
# inputB = Input(shape=(128,))
# # the first branch operates on the first input
# x = Dense(8, activation="relu")(inputA)
# x = Dense(4, activation="relu")(x)
# x = Model(inputs=inputA, outputs=x)
# # the second branch opreates on the second input
# y = Dense(64, activation="relu")(inputB)
# y = Dense(32, activation="relu")(y)
# y = Dense(4, activation="relu")(y)
# y = Model(inputs=inputB, outputs=y)
# # combine the output of the two branches
# combined = concatenate([x.output, y.output])
# # apply a FC layer and then a regression prediction on the
# # combined outputs
# z = Dense(2, activation="relu")(combined)
# z = Dense(1, activation="linear")(z)
# # our model will accept the inputs of the two branches and
# # then output a single value
# model = Model(inputs=[x.input, y.input], outputs=z)
