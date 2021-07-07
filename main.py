import tensorflow as tf
from Model import networkModel
from StringGenerator import stringGenerator
from configs import *
# Must implement loss function




model = networkModel()

XTrain, yTrain = getDataset(TRAINING_SIZE)
XTest, yTest = getDataset(TEST_SIZE)

model.compile(optimizer='adam', loss=lossFunction, metrics=['accuracy'])
model.fit(XTrain, yTrain, epochs=EPOCHS)

model.evaluate(XTest, yTest)
