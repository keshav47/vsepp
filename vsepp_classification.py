from comet_ml import Experiment
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
import yaml

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="2XZLX4rf67ICxldQrqBKVAFjT",
                        project_name="vsepp", workspace="iiitian-chandan")

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Add
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers as initializers, regularizers, constraints
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
import sys
import os
import math
import random
import warnings
import numpy as np
from sklearn import svm
import keras.backend as K
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input,Add
from keras.engine.topology import Layer
from sklearn.metrics import accuracy_score
from keras.layers.core import  Activation, Dense
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import optimizers
import numpy as np
import pandas as pd
import cv2
import keras
import keras.backend as K
from scipy import spatial
from statistics import mean
from tqdm import tqdm
import pickle

keys = ['parent_category','child_category']

EPOCHS = 100
BATCH_SIZE = 64
MODEL_OUTPUT_DIR = '/home/jupyter/filestore/keshav/vsepp/weights/classification/'

with open('/home/jupyter/filestore/keshav/vsepp/data/fashion/config_classification.pickle', 'rb') as handle:
    config = pickle.load(handle)

with open('/home/jupyter/filestore/keshav/vsepp/data/fashion/train_classification.pickle', 'rb') as handle:
    att_config_train = pickle.load(handle)

with open('/home/jupyter/filestore/keshav/vsepp/data/fashion/test_classification.pickle', 'rb') as handle:
    att_config_test = pickle.load(handle)

x_train = np.load("/home/jupyter/filestore/keshav/vsepp/data/fashion/image_embedding.npy")
x_test = np.load("/home/jupyter/filestore/keshav/vsepp/data/fashion/test_image_embedding.npy")


indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = np.asarray(x_train)[indices]


fit_list_train = []
for key in keys:
    fit_list_train.append(np.asarray(att_config_train[key])[indices])

fit_list_test = []
for key in keys:
    fit_list_test.append(np.array(att_config_test[key]))



nb_attributes = []

for key in keys:
    nb_attributes.append(config[key+"_no_of_labels"])

nb_classes = config[keys[0]+"_no_of_labels"]

base_output = Input(shape=(1024,))
first_dense_list = []
for i in range(len(keys)):
    l_mlp = Dense(128, activation='relu')(base_output)
    l_mlp = Dropout(0.5)(l_mlp)
    first_dense_list.append(l_mlp)

right_dense_list = []
left_dense_list = []
for i in range(len(keys)):
    l_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(first_dense_list[i])
    if i!=0:
        l_add = Add()([l_dense,dense_l_dense])
        dense_l = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l_add)
    else:
        dense_l = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l_dense)
    dense_l_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_l)
    left_dense_list.append(dense_l)

for i in reversed(range(len(keys))):
    r_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(first_dense_list[i])
    if i!=len(keys)-1:
        r_add = Add()([r_dense,dense_r_dense])
        dense_r = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(r_add)
    else:
        dense_r = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(r_dense)
    dense_r_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_r)
    right_dense_list.append(dense_r)

output_keys = [x.replace('&','').replace(' ','_') for x in keys]
output_list = []
for i in range(1,len(output_keys)+1):
    dense_l_output_add = Add()([left_dense_list[i-1],right_dense_list[-i]])
    dense_l_output = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_l_output_add)
    dense_l_output = Dropout(0.3)(dense_l_output)
    dense_l_output = Dense(nb_attributes[i-1], activation='softmax',name=output_keys[i-1]+'_output')(dense_l_output)
    output_list.append(dense_l_output)

loss = {}
for i in output_keys:
    loss[i+"_output"] = "categorical_crossentropy"


model = Model(input=base_output,
              outputs=output_list)

model.compile(optimizer=optimizers.SGD(lr=5e-4), metrics=['accuracy'],
              loss=loss)

filepath = MODEL_OUTPUT_DIR + "model_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True,
                             mode='auto', period=1)
checkpoints = [checkpoint]

model.fit(x_train, fit_list_train,validation_data=(x_test,fit_list_test)
,epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=checkpoints)
