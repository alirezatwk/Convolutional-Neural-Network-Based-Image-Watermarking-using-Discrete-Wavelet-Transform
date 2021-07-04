import tensorflow as tf
from Model import networkModel
from StringGenerator import stringGenerator
# Must implement loss function

TRAINING_SIZE, TEST_SIZE = 100000, 40000
EPOCHS = 10



model = networkModel()

XTrain, yTrain = getDataset(TRAINING_SIZE)
XTest, yTest = getDataset(TEST_SIZE)

model.compile(optimizer='adam', loss=lossFunction, metrics=['accuracy'])
model.fit(XTrain, yTrain, epochs=EPOCHS)

model.evaluate(XTest, yTest)
