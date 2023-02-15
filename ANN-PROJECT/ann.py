# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#IMPORT LIBRARIES
import keras
from keras_layer_normalization import LayerNormalization
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Convolution1D,BatchNormalization
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
import datetime
#IMPORT DATASET
x_train = np.genfromtxt(r'C:\\Users\\Lenovo\\Desktop\\x_train.txt',delimiter="")
y_train = np.genfromtxt(r'C:\\Users\\Lenovo\\Desktop\\y_train.txt',delimiter="")
x_test = np.genfromtxt(r'C:\\Users\\Lenovo\\Desktop\\x_test.txt',delimiter="")
y_test = np.genfromtxt(r'C:\\Users\\Lenovo\\Desktop\\y_test.txt',delimiter="")
#CHANGE LABELS SET TO ONE HOTVECTOR AND CHANGE TYPES OF TRAINING SET                   
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#RESHAPE ALL VECTORS TO ONE COLUMN
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#CREATE OUR MODEL IN KERAS
CNNLSTM_MODEL_SIMCHI = Sequential()
CNNLSTM_MODEL_SIMCHI.add(LSTM(30,return_sequences=True,activation = 'sigmoid',input_shape=(561,1)))
CNNLSTM_MODEL_SIMCHI.add(LayerNormalization())
CNNLSTM_MODEL_SIMCHI.add(BatchNormalization(momentum=0.99, epsilon=0.1))
CNNLSTM_MODEL_SIMCHI.add(Dropout(0.2))
CNNLSTM_MODEL_SIMCHIL.add(Convolution1D(16, (3), activation='relu'))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.BatchNormalization(momentum=0.99, epsilon=0.1))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.Dropout(0.2))
CNNLSTM_MODEL_SIMCHI.add(LSTM(30,return_sequences=True,activation = 'sigmoid'))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.BatchNormalization(momentum=0.99, epsilon=0.1))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.Dropout(0.2))
CNNLSTM_MODEL_SIMCHI.add(Convolution1D(32, (3), activation='relu'))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.BatchNormalization(momentum=0.99, epsilon=0.1))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.Dropout(0.2))
CNNLSTM_MODEL_SIMCHI.add(LSTM(30,return_sequences=False,activation = 'sigmoid'))
CNNLSTM_MODEL_SIMCHI.add(keras.layers.BatchNormalization(momentum=0.99, epsilon=0.1))
CNNLSTM_MODEL_SIMCHI.add(Dropout(0.2))
CNNLSTM_MODEL_SIMCHI.add(Dense(7, activation='softmax'))
#CHANGE NAMES 
CNNLSTM_MODEL_SIMCHI.name[0] = 'INPUT '
CNNLSTM_MODEL_SIMCHI.name[1] = 'HIDDEN 1 '
CNNLSTM_MODEL_SIMCHI.name[2]= 'HIDDEN 2 '
CNNLSTM_MODEL_SIMCHI.name[3]= 'HIDDEN 3 '
CNNLSTM_MODEL_SIMCHI.name[4]= 'HIDDEN 4 '
CNNLSTM_MODEL_SIMCHI.name[5]= 'HIDDEN 5 '
CNNLSTM_MODEL_SIMCHI.name[6]= 'HIDDEN 6 '
CNNLSTM_MODEL_SIMCHI.name[7]= 'HIDDEN 7 '
CNNLSTM_MODEL_SIMCHI.name[8]= 'HIDDEN 8 '
CNNLSTM_MODEL_SIMCHI.name[9]= 'HIDDEN 9 '
CNNLSTM_MODEL_SIMCHI.name[10]= 'HIDDEN 10 '
CNNLSTM_MODEL_SIMCHI.name[11]= 'HIDDEN 11 '
CNNLSTM_MODEL_SIMCHI.name[12]='HIDDEN 12 '
CNNLSTM_MODEL_SIMCHI.name[13]= 'HIDDEN 13 '
CNNLSTM_MODEL_SIMCHI.name[14]= 'HIDDEN 14 '
CNNLSTM_MODEL_SIMCHI.name[15]= 'HIDDEN 15 '
CNNLSTM_MODEL_SIMCHI.name[16]= 'HIDDEN 16 '
CNNLSTM_MODEL_SIMCHI.name[16]= 'OUTPUT '
#DRAW AND SAVE OUR MODEL
plot_model(CNNLSTM_MODEL_SIMCHI,to_file='HOSSEIN_SIMCHI_FINAL_PROJECT.pdf',show_shapes=True)
#COMPILE OUR MODEL
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#PRINT SUMMARY OF OUR MODEL
print(CNNLSTM_MODEL_SIMCHI.summary())
#FITTING ON OUR MODEL AND SHOW ELAPSED TIME
start = datetime.datetime.now()
network_history = CNNLSTM_MODEL_SIMCHI.fit(x_train,y_train, epochs=25, batch_size=200,validation_split=0.2)
end = datetime.datetime.now()
elapsed = end - start
print('Total training time : ',str(elapsed))
#EVALUATE OUR MODEL
test_loss,test_acc=CNNLSTM_MODEL_SIMCHI.evaluate(x_test,y_test)
print(test_loss)
print(test_acc)
#PLOT OUR MODEL ON DIAGRAM
history = network_history.history
losses = history['loss']
accuracies = history['acc']
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
val_losses = history['val_loss']
plt.plot(val_losses)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(accuracies)
val_accuracies = history['val_acc']
plt.plot(val_accuracies)
plt.legend(['acc','val_acc'])






