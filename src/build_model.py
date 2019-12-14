#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:35:39 2019

@author: jeremiah
"""

#%% importings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

#%%
def build_cnn_model(n_classes, input_shape, filter_size = 3, gray_scale = False, drop_prob_conv = 0.25, drop_prob_fully = 0.5):
    
    model = Sequential()
    
    model.add(Conv2D(4, (filter_size, filter_size), input_shape=input_shape, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(4, (filter_size, filter_size), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob_conv))
    
    model.add(Conv2D(8, (filter_size, filter_size), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(8, (filter_size, filter_size), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob_conv))

    model.add(Conv2D(12, (filter_size, filter_size), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(12, (filter_size, filter_size), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_prob_conv))

    model.add(Flatten())
    model.add(Dropout(drop_prob_fully))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(drop_prob_fully))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(drop_prob_fully))
    model.add(Dense(n_classes, activation="softmax"))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()
    
    return model
    