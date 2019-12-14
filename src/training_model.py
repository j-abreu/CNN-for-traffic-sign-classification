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
from keras import backend as K

sys.path.insert(1, '/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/src')

import build_model, get_data

#%% fixing seed

seed = 77
np.random.seed(seed)

#%% loading data

datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC_v2/Training"
img_size = 64
gray = False
categories = []

for folder in os.listdir(datadir):
    categories.append(folder)

X_train, y_train = get_data.make_data(datadir, categories, img_size, gray_scale=gray)

datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC_v2/Testing"
X_test, y_test = get_data.make_data(datadir, categories, img_size, gray_scale=gray)

#%% preparing data
#X_train = np.array(X_train/255.0, "float32")
#X_test = np.array(X_test/255.0, "float32")

X_test, X_hold, y_test, y_hold = train_test_split(X_test, y_test, test_size=0.3, shuffle=False, random_state=seed)

#%%
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_hold = np_utils.to_categorical(y_hold)

#%%
n_classes = y_train.shape[1]

model = build_model.build_cnn_model(n_classes, X_train.shape[1:], drop_prob_conv=0.25, drop_prob_fully=0.5)


#%%
batch_size = 20
epochs = 50
#%%

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

#%% loading and compiling model
path = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/checkpoint_aug2.json"
json_file = open(path, "r")
loaded_model = json_file.read()
json_file.close()

model = model_from_json(loaded_model)

model.load_weights("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/weights_aug2.hdf5")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
model.summary()

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
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Test"], loc="best")
plt.savefig("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/figures/model_loss.png")
plt.show()

#%%

img_array = cv2.imread("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/4.jpg")
img_array = cv2.resize(img_array, (img_size, img_size))
img_array = np.array(img_array, "float32").reshape(1, img_size, img_size, 3)

layer = 3
get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
layer_output = get_layer_output([img_array])[0]

layer_output = np.array(layer_output, "float32").reshape(layer_output.shape[1], layer_output.shape[2],
                       layer_output.shape[3])

#%%

x = 0

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.imshow(layer_output[:,:,0+x])
plt.subplot(222)
plt.imshow(layer_output[:,:,1+x])
plt.subplot(223)
plt.imshow(layer_output[:,:,2+x])
plt.subplot(224)
plt.imshow(layer_output[:,:,3+x])

plt.show()


#%%

results = model.evaluate(X_hold, y_hold, batch_size=1)
print("Accuracy: {:.2f}%".format(results[1]*100))


#%%

path = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/"
plt.figure(figsize=(15,12))
i = 0
for img in os.listdir(path):
    i += 1
    img_array = cv2.imread(os.path.join(path, img))
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = np.array(img_array, "float32").reshape(-1, img_size, img_size, 3)
    
    
    prediction = model.predict(img_array)
    idx = tf.argmax(prediction, axis=1)
    s = str(categories[int(idx)])
    s += "\n" + str(max(max(prediction)))
    img_array = img_array.reshape(img_size, img_size, 3)
    img_array = img_array/255.0
    plt.subplot(3, 5, i)
    plt.imshow(img_array)
    plt.title(s)

plt.show()

#%%
    
    
    

#%%

model_json = model.to_json()

with open("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/checkpoint_aug2.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/weights_aug2.hdf5")


















