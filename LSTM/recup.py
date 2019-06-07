import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from IPython.display import Image, display
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from datetime import datetime
from keras.callbacks import TensorBoard
from keras.layers import multiply, add
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from skimage import io, viewer

train_csv = pd.read_csv('../Data/train.csv')
print(train_csv.describe())
print(train_csv.head())

filenames = ['../Data/train/' + fname for fname in train_csv['id'].tolist()]
labels = train_csv['has_cactus'].tolist()

train=[]
for file_name in filenames:
    img=cv2.imread(file_name)
    img=img.reshape(32*32*3,)
    img=img/255
    train.append(img)

x_train, x_test, y_train, y_test = train_test_split(train,
                                                    labels,
                                                    train_size=0.9)

train = np.array(x_train)
train = np.reshape(train, (-1, 32, 32, 3))
print(train.shape)
plt.imshow(train[1])
