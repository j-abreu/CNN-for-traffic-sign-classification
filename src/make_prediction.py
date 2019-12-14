#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:29:13 2019

@author: jeremiah
"""

#%%
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import cv2
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from keras import backend as K

sys.path.insert(1, '/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/src')


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
datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC/Training"

categories = []

for folder in os.listdir(datadir):
    categories.append(folder)
label = ['dead_end', 'dead_end', 'stop', 'no_right_turn', 'no_right_turn',
         'parking_allowed', 'priority_road', 'priority_road', 'speed_bump',
         'stop', 'dead_end', 'parking_allowed', 'no_entry_for_all_drivers', 'no_parking',
         'no_right_turn']
#%%
img_size = 64
path = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/"
plt.figure(figsize=(15,12))
i = 0
for img in os.listdir(path):
    #plt.figure(figsize=(12, 12))
    img_array = cv2.imread(os.path.join(path, img))
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = np.array(img_array, "float32").reshape(-1, img_size, img_size, 3)
    
    
    prediction = model.predict(img_array)
    idx = tf.argmax(prediction, axis=1)
    s = str(categories[int(idx)])
    if s == label[i]:
        color = 'g'
    else:
        color = 'r'
    s += "\n" + str(max(max(prediction)))
    s += "\n(" + label[i] + ")"
    img_array = img_array.reshape(img_size, img_size, 3)
    img_array = img_array/255.0
    plt.subplot(3, 5, i+1)
    i += 1
    plt.imshow(img_array)
    plt.title(s, color = color)
plt.savefig("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/figures/testing_model.png")
plt.show()

#%%
layers = [0, 3, 5, 7, 10, 12, 14, 17, 19]

for layer in layers:
    img_array = cv2.imread("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/3.png")
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = np.array(img_array, "float32").reshape(1, img_size, img_size, 3)
    
    get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[layer].output])
    layer_output = get_layer_output([img_array])[0]
    
    layer_output = np.array(layer_output, "float32").reshape(layer_output.shape[1], layer_output.shape[2],
                           layer_output.shape[3])
    
    
    x = 0
    print("layer ", layer)
    
    plt.figure(figsize=(12,12))
    plt.subplot(221)
    plt.imshow(layer_output[:,:,0+x], cmap="winter")
    plt.subplot(222)
    plt.imshow(layer_output[:,:,1+x], cmap="winter")
    plt.subplot(223)
    plt.imshow(layer_output[:,:,2+x], cmap="winter")
    plt.subplot(224)
    plt.imshow(layer_output[:,:,3+x], cmap="winter")
    plt.show()













