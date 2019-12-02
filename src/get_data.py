#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:31:27 2019

@author: jeremiah
"""

#%% importings
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

#%% 
def make_data_color(datadir, categories, img_size):
    """
    This function take a dataset of images and transform to a numpy array and save it.
    :param datadir: path to the dataset.
    :param categories: list of categories as String.
    :param img_size: size of the image in the format img_size by img_size.
    :return: the X and y numpy array. (saves to the path too)
    """
    training_data = []
    
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Error in reading data")
    
    print("Data readed!\nLength of the dataset: ", len(training_data))
    
    random.shuffle(training_data)
    
    X = []
    y = []
    
    for features, label in training_data:
        X.append(features)
        y.append(label)
    
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    
    np.save(datadir+"_dataset", training_data)
    
    return X, y


def make_data_gray(datadir, categories, img_size):
    """
    This function take a dataset of images and transform to a numpy array and save it.
    :param datadir: path to the dataset.
    :param categories: list of categories as String.
    :param img_size: size of the image in the format img_size by img_size.
    :return: the X and y numpy array. (saves to the path too)
    """
    training_data = []
    
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Error in reading data")
    
    print("Data readed!\nLength of the dataset: ", len(training_data))
    
    random.shuffle(training_data)
    
    X = []
    y = []
    
    for features, label in training_data:
        X.append(features)
        y.append(label)
    
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    
    np.save(datadir+"_dataset", training_data)
    
    return X, y    
    
    
def make_data(datadir, categories, img_size, gray_scale = False):
    if gray_scale:
        X, y = make_data_gray(datadir, categories, img_size)
        return np.array(X, "float32"), np.array(y, "float32")
    else:
        X, y = make_data_color(datadir, categories, img_size)
        return np.array(X, "float32"), np.array(y, "float32")
    
    
    
    
    
    
    
    
    
    
    
    
    
    