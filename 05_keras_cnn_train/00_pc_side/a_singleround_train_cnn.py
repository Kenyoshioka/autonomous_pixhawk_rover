#!/usr/bin/env python3
# -*- coding: utf-8 -*-  

# common modules
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import json
import time
import csv
import cv2
import sys
import os

# machine learning common modules
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Reshape
from keras.models import Sequential
from sklearn import preprocessing
from keras import regularizers

# machine learning model visualization modules
from keras.utils import plot_model
import matplotlib.pyplot as plt

# preparation sequence
def preparation_sequence():

    # check os compatibility
    if os.name != 'nt':

        print ("\nThis codes only compatible with Windows OS!")
        exit()

    # check python environment compatibility
    if sys.version_info[0] < 3:

        print ("\nThis codes only compatible with Python 3 environment!")
        exit()

    # if folder0 doesn't exist end the program
    if not os.path.exists(folder0):

        print ("\nFolder 00_training_data, where the training data supposedly located does not exist!")
        exit()

    # create folder1 if it doesn't exist create it
    if not os.path.exists(folder1):

        os.makedirs(folder1)

# getch class for windows
class _GetchWindows:

    def __init__(self):

        import msvcrt

    def __call__(self):

        import msvcrt
        return msvcrt.getch()

# assign folders name
folder0 = "00_training_data"
folder1 = "01_model_weight"

# specify naming based on current time
timestr = time.strftime("%Y%m%d%H%M%S")

# execute preparation sequence
preparation_sequence()

# getch initialization
getch = _GetchWindows()

# loop until user choose prefered optimizer
while True:

    # clear screen
    os.system("cls")

    # choose your preference machine learning optimizer
    print ("Choose you preferred optimizer to use during training (The best one is Adadelta).")
    print ("The optimizer were set with default parameters, to change it edit the code.")
    print ("To quit press [q].\n")
    print (" [1] = SGD")
    print (" [2] = Adam")
    print (" [3] = Nadam")
    print (" [4] = Adamax")
    print (" [5] = RMSProp")
    print (" [6] = Adagard")
    print (" [7] = Adadelta\n")

    # get keyboard input
    ml_optimizer = getch()

    # when [q] is pressed
    if (ml_optimizer == b'q'):

        # clear screen
        os.system("cls")
        # quit the program
        exit()

    # when the number entered is [1] to [7] 
    elif (ml_optimizer > b'0' and ml_optimizer <= b'7'):

        # break while loop
        break

# clear screen
os.system("cls")

# print instruction
print ("To quit anytime press [CTRL] + [C].\n")

# get the start time when loading the data
time_load_start = time.time()

# print action that will be executed
print ("Load images and tuple from .npz to be trained")

# load jpg images to get image_array
training_images = glob.glob(folder0 + "/*.jpg")
image_array = np.array([cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in tqdm(training_images, desc = "Image")], dtype =  np.float64) # progress bar

# Load .npz to get label_array
training_data = glob.glob(folder0 + "/*.npz")
label_array = None
# unpacking the .npz content
for single_npz in training_data: # single_npz == one array representing one array of saved user input label for that image
        
    with np.load(single_npz) as data:
        
        train_labels_temp = data["train_labels"] # returns the training user input data array assigned to 'train_labels' argument
    
    label_array = np.array([label for label in tqdm(train_labels_temp, desc = "Array")], dtype=np.float64) # progress bar

# save image_array and label_array into X and y
X = image_array
y = label_array

# get the end time after loading the data
time_load_end = time.time()
time_load_total = time_load_end - time_load_start
print ("» Total time taken to load all the data: %.2f" % float(time_load_total), "[s]")

# normalize from 0 to 1
X = X / 255.

# print sample and test parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
print ("\nNumber of sample and test data for this training")
print ("» %s sample images with %s pixels of height and %s pixel of width" % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
print ("» %s test image with %s pixels of height and %s pixel of width" % (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
print ("» %s of sample input saved in array size of %s" % (y_train.shape[0], y_train.shape[1]))
print ("» %s of test input saved in array size of %s\n" % (y_test.shape[0], y_test.shape[1]))

# get the start time when training the CNN
time_training_start = time.time()

# initialising the CNN
model = Sequential()

# using TensorFlow backend
model.add(Reshape((120, 320, 1), input_shape = (120, 320), name = "input"))

# first convolutional layer and pooling
model.add(Conv2D(32, (5, 5), padding = "same", activation = "relu", name = "1st_cnn_layer"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# second convolutional layer and pooling (default parameter for regularizer: kernel_regularizer = regularizers.l2(0.01))
model.add(Conv2D(32, (5, 5), kernel_regularizer = regularizers.l2(0.0001), activation = "relu", name = "2nd_cnn_layer"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# third convolutional layer and pooling (default parameter for regularizer: activity_regularizer = regularizers.l1(0.01))
model.add(Conv2D(32, (5, 5), activity_regularizer = regularizers.l1(0.00001), activation = "relu", name = "3rd_cnn_layer"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# flatten, fully connected layer 1 (14,208 to 28,416 nodes)
model.add(Flatten(name = "1st_fully_connected_layer"))

# fully connected layer 2 (300 nodes)
model.add(Dense(300, kernel_initializer = "uniform", name = "2nd_fully_connected_layer"))
model.add(Dropout(0.20))
model.add(Activation("relu"))

# output layer (3 nodes)
model.add(Dense(3, kernel_initializer = "uniform", name = "output_layer"))
model.add(Activation("softmax"))

# sgd type of optimization
if (ml_optimizer == b'1'):

    print ("SGD optimizer selected.\n")
    from keras.optimizers import SGD
    sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss = "categorical_crossentropy",
                optimizer = sgd,
                metrics = ["accuracy"])

# adam type of optimization
elif (ml_optimizer == b'2'):

    print ("Adam optimizer selected.\n")
    from keras.optimizers import Adam
    adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    model.compile(loss = "categorical_crossentropy",
                optimizer = adam,
                metrics = ["accuracy"])

# nadam type of optimization
elif (ml_optimizer == b'3'):

    print ("Nadam optimizer selected.\n")
    from keras.optimizers import Nadam
    nadam = Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)
    model.compile(loss = "categorical_crossentropy",
                optimizer = nadam,
                metrics = ["accuracy"])

# adamax type of optimization
elif (ml_optimizer == b'4'):

    print ("Adamax optimizer selected.\n")
    from keras.optimizers import Adamax
    adamax = Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0)
    model.compile(loss = "categorical_crossentropy",
                optimizer = adamax,
                metrics = ["accuracy"])

# rmsprop type of optimization
elif (ml_optimizer == b'5'):

    print ("RMSProp optimizer selected.\n")
    from keras.optimizers import RMSprop
    rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon = None, decay = 0.0)
    model.compile(loss = "categorical_crossentropy",
                optimizer = rmsprop,
                metrics = ["accuracy"])

# adagard type of optimization
elif (ml_optimizer == b'6'):

    print ("Adagard optimizer selected.\n")
    from keras.optimizers import Adagrad
    adagard = Adagrad(lr = 0.01, epsilon = None, decay = 0.0)
    model.compile(loss = "categorical_crossentropy",
                optimizer = adagard,
                metrics = ["accuracy"])

# adadelta type of optimization
elif (ml_optimizer == b'7'):

    print ("Adadelta optimizer selected.\n")
    from keras.optimizers import Adadelta
    adadelta = Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
    model.compile(loss = "categorical_crossentropy", 
                        optimizer = adadelta,
                        metrics = ["accuracy"])

# keras_tf cnn model overview
plot_model(model, to_file = folder1 + "/model_overview.png", show_shapes = True, show_layer_names = True)

# set callback functions to early stop training
callbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 0, mode = 'auto'),
             ModelCheckpoint(filepath = folder1 + "/" + "{}.h5".format(timestr), monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto', period = 1)]

# fit the model
history = model.fit(X_train, y_train,
                    epochs = 20,
                    batch_size = 20,
                    callbacks = callbacks,
                    validation_data = (X_test, y_test))

# get the end time after training the CNN
time_training_end = time.time()
time_training_total = time_training_end - time_training_start
print ("\nTotal time taken to train model: ", int(time_training_total), "[s]")
print ("The best model + weight with the lowest val_loss is saved at /%s/%s.h5." % (folder1, timestr))

# save model only to json file
json_string = model.to_json()
with open(folder1 + "/" + "{}.json".format(timestr), "w") as new_json:

    json.dump(json_string, new_json)

# plot training and validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('History for model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.savefig(folder1+ "/model_accuracy.png")
plt.show()

# plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('History for model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.savefig(folder1 + "/model_loss.png")
plt.show()