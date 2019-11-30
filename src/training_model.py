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
import cv2
import tensorflow as tf

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

model_check = ModelCheckpoint("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/weights.hdf5",
                              monitor="val_acc", verbose=1, save_best_only=True, mode="max")

early_stop = EarlyStopping(monitor="val_acc", patience=10)

#%%
batch_size = 32
epochs = 20


history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                    callbacks=[model_check, early_stop], verbose=2)

#%%
print(history.history.keys())

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Test"], loc="best")
plt.savefig("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/figures/model_acc.png")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Test"], loc="best")
plt.savefig("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/figures/model_loss.png")
plt.show()

#%%

results = model.evaluate(X_hold, y_hold, batch_size=batch_size)
print("Accuracy: {:.2f}%".format(results[1]*100))


#%%

model_json = model.to_json()

with open("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/checkpoint.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/weights.hdf5")




#%%

img_array = cv2.imread("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/9.png")
img_array = cv2.resize(img_array, (img_size, img_size))
img_array = np.array(img_array, "float32").reshape(-1, img_size, img_size, 3)
img_array = img_array/255.0
#%%

prediction = model.predict(img_array)

idx = tf.argmax(prediction, axis=1)
print(categories[int(idx)])






















