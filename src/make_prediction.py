#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:29:13 2019

@author: jeremiah
"""

#%%
from keras.models import model_from_json
import tensorflow as tf
import cv2
import numpy as np
import os
import sys

sys.path.insert(1, '/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/src')

import build_model

#%% loading and compiling model
path = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/checkpoint.json"
json_file = open(path, "r")
loaded_model = json_file.read()
json_file.close()

model = model_from_json(loaded_model)

model.load_weights("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/CNN-for-traffic-sign-classification/trained_models/weights.hdf5")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
model.summary()

#%%
datadir = "/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/BelgiumTSC/Training"

categories = []

for folder in os.listdir(datadir):
    categories.append(folder)
#%%

img_size = 64

img_array = cv2.imread("/media/jeremiah/7E9BF5A34D96B6A4/2019.4/PE3/teste/02434_00001.ppm")
img_array = cv2.resize(img_array, (img_size, img_size))
img_array = np.array(img_array, "float32").reshape(img_size, img_size, 3)
img_array = img_array/255.0


#%%

res = model(img_array)

#%%

prediction = model.predict(img_array)

idx = tf.argmax(prediction, axis=1)
print(categories[int(idx)])


#%%

















