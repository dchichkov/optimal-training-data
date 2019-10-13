''' 
Insight.  Data is the new source code.  Applying evolutionary or other 
optimization algoritms to 'optimize' that source code is interesting.
Goals of optimization - increase accuracy, training spead, etc.

Low hanging fruits are:
 * optimal subsampling of datasets;
 * optimizing datasets for single-epoch training;
 * optimizing datasets for finetuning;

Goal: Subsample the MNIST training set to provide minimal and optimal set of examples 
that maximizes a test score of an arbitrary neural network trained on that set.

Example: Code below evolves indexes (a subset of 4096 samples from MNIST dataset) 
to optimize validation accuracy of a simple convnet trained for 1 epochs on that subset.

It trains another convnet (one extra layer of convolutions) on the same indexes 
and evaluates its accuracy on the test set, to get the *test accuracy*.

If the 'optimal' dataset is somewhat general between architecures, we should
see the increase of test accuracy during evolution of the indexes.
'''

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras
core_config = tf.ConfigProto()
core_config.gpu_options.allow_growth = False
core_config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=core_config)
keras.backend.set_session(session)

import random, numpy, collections, time
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# The number of epochs to train and batch size
epochs = 1
batch_size = 32

# input image dimensions, number of classes
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create a training set - 4096 out of 50%, validation set - 50%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.50, random_state=30)

print('x_train shape:', x_train.shape, 'x_test.shape', x_test.shape)

def MNIST(is_test = False):
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Flatten
  from keras.layers import Conv2D, MaxPooling2D

  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  # changing architecture for test model
  if is_test:
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  return model

def f(indexes, is_test = False):
    """
      input: indexes to train the model on
      is_test:
        false: returns 'training' model accuracy on the validation set
        true: returns 'test' model accuracy on the test set
    """

    global x_test, y_test, x_train, y_train, x_val, y_val, step
    model = MNIST(is_test)
    # print("len(indexes)", len(indexes), indexes[:10])
    
    hist = model.fit(x_train[indexes], y_train[indexes],
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=((x_val, y_val)))
    validation_acccuracy = list(hist.history['val_acc'])[-1]

    # This is used during evaluation only
    if is_test:
      score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
      return score[1]

    return (validation_acccuracy,)


#for i in range(100):
#    print("Initial accuracy:", f(range(4096)))
#print("Initial Test accuracy:", f(range(4096), is_test = True))
# initial = model.get_weights()   # initial trained parameters


