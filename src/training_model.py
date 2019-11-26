#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:14:44 2019

@author: jeremiah
"""

#%% importings
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

sys.path.insert(1, '/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/src')

import build_model, get_data

#%% fixing seed

seed = 77
np.random.seed(seed)

#%% loading data

datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC/Training"
img_size = 64
gray_scale = False
categories = []

for folder in os.listdir(datadir):
    categories.append(folder)

X_train, y_train = get_data.make_data(datadir, categories, img_size)

datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC/Testing"
X_test, y_test = get_data.make_data(datadir, categories, img_size)

#%% preparing data
X_train = np.array(X_train/255.0, "float32")
X_test = np.array(X_test/255.0, "float32")

X_test, X_hold, y_test, y_hold = train_test_split(X_test, y_test, test_size=0.3, shuffle=False, random_state=seed)

#%%
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_hold = np_utils.to_categorical(y_hold)

#%%
n_classes = y_train.shape[1]

model = build_model.build_cnn_model(n_classes, X_train.shape[1:], drop_prob = 0.2)

model_check = ModelCheckpoint("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models.hdf5",
                              monitor="val_acc", verbose=1, save_best_only=True, mode="max")

early_stop = EarlyStopping(monitor="val_acc", patience=20)

#%%
batch_size = 32
epochs = 200


history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                    callbacks=[model_check, early_stop], verbose=2)

#%%



















