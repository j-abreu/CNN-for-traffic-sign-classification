#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:35:39 2019

@author: jeremiah
"""

#%% importings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

#%%
def build_cnn_model(n_classes, input_shape, filter_size = 3, gray_scale = False, drop_prob = 0.0):
    
    model = Sequential()
    
    model.add(Conv2D(16, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob))
    
    model.add(Conv2D(32, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob))
    
    model.add(Conv2D(64, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (filter_size, filter_size), input_shape=input_shape, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob))
    
    
    model.add(Flatten())
    model.add(Dropout(drop_prob))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(drop_prob))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(drop_prob))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(drop_prob))
    model.add(Dense(n_classes, activation="softmax"))
    
    model.compile(optimizer=Adam(lr=0.0004), loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()
    
    return model
    