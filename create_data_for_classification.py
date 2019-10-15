import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import sys
import os
import yaml
from tqdm import tqdm


from keras.utils.np_utils import to_categorical
import pickle

dataFrame = pd.read_csv("/home/jupyter/filestore/keshav/vestiairecollective/data/vestiairecollective_train.csv")
dataFrame = dataFrame[['meta.training_ir_attributes.category.l1','meta.training_ir_attributes.category.l2','meta.training_ir_attributes.category.l3']]
print(len(dataFrame))
count = len(dataFrame)
dataFrame_test = pd.read_csv("/home/jupyter/filestore/keshav/vestiairecollective/data/vestiairecollective_test.csv")
dataFrame_test = dataFrame_test[['meta.training_ir_attributes.category.l1','meta.training_ir_attributes.category.l2','meta.training_ir_attributes.category.l3']]
print(len(dataFrame_test))
dataFrame = dataFrame.append(dataFrame_test,ignore_index=True)
dataFrame = dataFrame.replace(np.nan, '', regex=True)

keys = ['meta.training_ir_attributes.category.l1','meta.training_ir_attributes.category.l2','meta.training_ir_attributes.category.l3']

def initialize_dict(keys):
    main_dict = {}
    for key in keys:
        main_dict[key+"_labels_encoding"] = {}
        main_dict[key+"_labels_distribution"] = {}
        main_dict[key+"_no_of_labels"] = 0
    return main_dict

config = initialize_dict(keys)

def get_data():
    return_list = []
    temp_full_dict = {}
    for key in keys:
        temp_full_dict[key] = dataFrame[key].tolist()
    for i in range(len(dataFrame)):
        temp_dict = {}
        for key in keys:
            temp_dict[key] = temp_full_dict[key][i]
        return_list.append(temp_dict)
    return return_list

data = get_data()

df = {}
for i in keys:
    df[i] = []


def encode_labels(label, level):
    if label == "":
        label = "null"

    if label not in config[level+"_labels_encoding"].keys():
        config[level+"_labels_encoding"][label] = config[level+"_no_of_labels"]
        config[level+"_labels_distribution"][label] = 0
        config[level+"_no_of_labels"] += 1
    config[level+"_labels_distribution"][label] += 1
    return config[level+"_labels_encoding"][label]



for d in tqdm(data):
    for key in keys:
        df[key].append(encode_labels(d[key], key))

save_dict = {}
for key in keys:
    save_dict[key] = to_categorical(np.asarray(df[key]))

test_save_dict = {}
test_save_dict['meta.training_ir_attributes.category.l1'] = save_dict['meta.training_ir_attributes.category.l1'][count:]
test_save_dict['meta.training_ir_attributes.category.l2'] = save_dict['meta.training_ir_attributes.category.l2'][count:]
test_save_dict['meta.training_ir_attributes.category.l3'] = save_dict['meta.training_ir_attributes.category.l3'][count:]
train_save_dict = {}
train_save_dict['meta.training_ir_attributes.category.l1'] = save_dict['meta.training_ir_attributes.category.l1'][:count]
train_save_dict['meta.training_ir_attributes.category.l2'] = save_dict['meta.training_ir_attributes.category.l2'][:count]
train_save_dict['meta.training_ir_attributes.category.l3'] = save_dict['meta.training_ir_attributes.category.l3'][:count]


with open('/home/jupyter/filestore/keshav/vestiairecollective/data/vestiairecollective_classification_train.pickle', 'wb') as handle:
    pickle.dump(train_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/jupyter/filestore/keshav/vestiairecollective/data/vestiairecollective_classification_test.pickle', 'wb') as handle:
    pickle.dump(test_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/jupyter/filestore/keshav/vestiairecollective/data/vestiairecollective_classification_config.pickle', 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(config)
