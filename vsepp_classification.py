# from comet_ml import Experiment
import numpy as np
import pandas as pd
import pickle


keys = ['parent_category','child_category']

fit_list_train = []
for key in keys:
    fit_list_train.append(np.array(train_save_dict[key]))

fit_list_test = []
for key in keys:
    fit_list_test.append(np.array(test_save_dict[key]))


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

model.compile(optimizer=optimizers.SGD(lr=1e-2), metrics=['accuracy'],
              loss=loss)

# filepath = MODEL_OUTPUT_DIR + "model_{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True,
#                              mode='auto', period=1)
# checkpoints = [checkpoint]
EPOCHS = 100
BATCH_SIZE = 50

model.fit(x_train, fit_list_train,validation_data=(x_test,fit_list_test)
,epochs=EPOCHS, batch_size=BATCH_SIZE)
