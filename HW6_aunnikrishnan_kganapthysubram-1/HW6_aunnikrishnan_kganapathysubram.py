import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import time
import math
import itertools
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
import keras
from keras.callbacks import ModelCheckpoint

trainY = np.load("ages.npy")
trainX = np.load("faces.npy")/255

def shuffle(x, y):
    n = y.shape[0]
    # print("shuffle", n)
    shuffle_idx = np.random.permutation(n)
    new_x = x[shuffle_idx]
    new_y = y[shuffle_idx]
    # print(new_x.shape)

    return new_x, new_y


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


trainX, trainY = shuffle(trainX, trainY)

gg = np.stack((trainX,)*3, axis = -1)
(x_train, x_test) = gg[:6000], gg[6000:]
(y_train, y_test) = trainY[:6000], trainY[6000:]
(x_train, x_valid) = x_train[:5400], x_train[5400:]
(y_train, y_valid) = y_train[:5400], y_train[5400:]

model = VGG16(input_shape=[48,48,3],weights='imagenet', include_top=False)

x=Flatten()(model.output)
x=Dense(256,activation='relu')(x)
x=Dense(128,activation='relu')(x)
prediction=Dense(1)(x)
final_model=Model(inputs=model.input,outputs=prediction)
final_model.summary()

final_model.compile(optimizer = "adam", loss = root_mean_squared_error,
              metrics =["accuracy"])

checkpointer = ModelCheckpoint(filepath='final_model.weights.best.hdf5', verbose=1, save_best_only=True)
final_model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=20,
          validation_data=(x_valid, y_valid),
          callbacks=[checkpointer])

final_model.load_weights('final_model.weights.best.hdf5')
score = final_model.evaluate(x_test, y_test, verbose=0)
print(score)