import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#################################################################
# Insert TensorFlow code here to complete the tutorial in part 1.
#################################################################
# Load the Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Normalize the Data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_valid, y_valid),
          callbacks=[checkpointer])

# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy_ For COLLAB CODE:', score[1])

'''y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))'''

#################################################################
# Insert TensorFlow code here to *train* the CNN for part 2.
#################################################################

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2)))
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_valid, y_valid),
          callbacks=[checkpointer])

model.load_weights('model.weights.best.hdf5')

yhat1 = model.predict(x_train[0:1, :, :, :])[0]  # Save model's output

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy for professor's suggested NN
print('\n', 'Test accuracy_OUR IMPLEMENTATION:', score[1])

'''y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))'''


#################################################################
# Write a method to extract the weights from the trained
# TensorFlow model. In particular, be *careful* of the fact that
# TensorFlow packs the convolution kernels as KxKx1xF, where
# K is the width of the filter and F is the number of filters.
#################################################################

# Function to extract W1
def create_w1(x):
    W1_ = np.zeros((64, 26 * 26, 28 * 28))
    k = 0
    for kk in range(64):
        # print("weight:", kk)
        alpha = np.zeros((28, 28))
        filter = x[:, :, kk]
        # print(filter)
        filter_size = filter.shape[0]
        alpha[:filter.shape[0], :filter.shape[1]] = filter
        alpha = alpha.reshape(-1, 1)
        alpha = alpha.T
        temp = alpha
        for k in range(k, k + 1):
            # print(k)
            alpha = temp
            for i in range(0, W1_.shape[1]):
                W1_[k, i, :] = alpha
                alpha = np.roll(alpha, 1)
            for j in range(0, W1_.shape[1], 26):
                # print(j)
                if j != 0:
                    W1_[k, j:, :] = np.roll(W1_[k, j:, :], (filter_size - 1))
        k = k + 1
        # print(k)
    return W1_


# Function to extract b1
def repeat(arr, count):
    return np.stack([arr for _ in range(count)], axis=0)


# Function to conver weights from keras to our FCNN implementation
def convertWeights(model):
    # Extract W1, b1, W2, b2, W3, b3 from model.
    # ...
    W1 = model.get_weights()[0]
    W1 = W1.reshape(3, 3, 64)
    lo = create_w1(W1)
    W1 = lo.reshape((43264, 784), order="F")
    b1 = repeat(model.get_weights()[1], 676)
    # print(b1.shape)
    # print(b1.flatten())
    b1 = b1.flatten()
    # print(b1.shape)
    W2 = model.get_weights()[2]
    b2 = model.get_weights()[3]
    W3 = model.get_weights()[4]
    b3 = model.get_weights()[5]

    return W1, b1, W2, b2, W3, b3


#################################################################
# Below here, use numpy code ONLY (i.e., no TensorFlow) to use the
# extracted weights to replicate the output of the TensorFlow model.
#################################################################

# Implement a fully-connected layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def fullyConnected(W1, b1, W2, b2, W3, b3, x):
    h = np.matmul(W1, x) + b1
    h = h.reshape((64, 26, 26), order="F")
    mp = maxpooling(h)
    mp = mp.reshape((13 * 13 * 64), order="F")
    fc1 = relu(mp)
    fc1 = relu(np.matmul(fc1.T, model.get_weights()[2]) + model.get_weights()[3])
    fc2 = np.matmul(fc1.T, model.get_weights()[4]) + model.get_weights()[5]
    op = softmax(fc2)
    # returns Softmax of our predictions
    return op


# Implement a max-pooling layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def maxpooling(arr):
    filter_size = 2
    stride = 2
    max_pooled = np.zeros((64, 13, 13))
    # print(max_pooled.shape)
    for k in range(arr.shape[0]):
        m = 0
        for i in range(0, arr.shape[2], stride):
            l = 0
            for j in range(0, arr.shape[1], stride):
                # print(i,j)
                if stride == 1:
                    if j == max_pooled.shape[1] or i == max_pooled.shape[2]:
                        print("break")
                        break
                    else:
                        if j + 1 >= arr.shape[2] or i + 1 >= arr.shape[1]:
                            print("break1")
                            break
                        else:
                            conv_j = arr[k, i:i + filter_size, j:j + filter_size]
                            print(conv_j)
                            if conv_j.size == 0:
                                break
                            else:
                                conv_j = conv_j.reshape(-1, 1)
                                index = np.argmax(conv_j)
                                max_pooled[k, i - m, j - l] = conv_j[index]
                else:
                    if j + 1 >= arr.shape[2] or i + 1 >= arr.shape[1]:
                        print("break1")
                        break
                    else:
                        conv_j = arr[k, i:i + filter_size, j:j + filter_size]
                        # print(conv_j)
                        if conv_j.size == 0:
                            break
                        else:
                            conv_j = conv_j.reshape(-1, 1)
                            index = np.argmax(conv_j)
                            max_pooled[k, i - m, j - l] = conv_j[index]
                            l = l + stride - 1
            m = m + stride - 1

    return max_pooled


# Implement a Relu function.
def relu(x):
    # return np.maximum(0,x)
    x_ = x.copy()
    x_[x_ < 0] = 0

    return x_


# Implement a Softmax function.
def softmax(z):
    term_1 = np.exp(z)
    denom = np.sum(term_1, axis=0, keepdims=True)
    return term_1 / denom


# Load weights from TensorFlow-trained model.
W1, b1, W2, b2, W3, b3 = convertWeights(model)

# Implement the CNN with the same architecture and weights
# as the TensorFlow-trained model but using only numpy.
# yhat2 = softmax(...)
# Taking example image for testing. Same image as taken for yhat1
x = x_train[0:1, :, :, :]
x = x.flatten()

yhat2 = fullyConnected(W1, b1, W2, b2, W3, b3, x)
print("This is the softmax output of Keras implementation", yhat1)
print("This is the output of our implementation", yhat2)
