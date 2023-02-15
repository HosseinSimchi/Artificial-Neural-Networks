# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:58:26 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#IMPORT LIBRARIES
from keras_layer_normalization import LayerNormalization
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
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
LSTM_MODEL_SIMCHI = Sequential()
LSTM_MODEL_SIMCHI.add(LSTM(32, return_sequences=True,input_shape=(x_train.shape[1],1)))
LSTM_MODEL_SIMCHI.add(LayerNormalization())
LSTM_MODEL_SIMCHI.add(LSTM(32, return_sequences=True,activation='tanh'))
LSTM_MODEL_SIMCHI.add(Dropout(0.2))
LSTM_MODEL_SIMCHI.add(LSTM(32))
LSTM_MODEL_SIMCHI.add(Dense(7, activation='softmax'))
#CHANGE NAMES 
LSTM_MODEL_SIMCHI.layers[0].name = 'INPUT '
LSTM_MODEL_SIMCHI.layers[1].name = 'HIDDEN 1 '
LSTM_MODEL_SIMCHI.layers[2].name= 'HIDDEN 2 '
LSTM_MODEL_SIMCHI.layers[3].name= 'HIDDEN 3 '
LSTM_MODEL_SIMCHI.layers[4].name= 'HIDDEN 4 '
LSTM_MODEL_SIMCHI.layers[5].name= 'OUTPUT '
#DRAW AND SAVE OUR MODEL
plot_model(LSTM_MODEL_SIMCHI,to_file='HOSSEIN_SIMCHI_LSTM_PROJECT1.pdf',show_shapes=True)
#COMPILE OUR MODEL
LSTM_MODEL_SIMCHI.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#PRINT SUMMARY OF OUR MODEL
print(LSTM_MODEL_SIMCHI.summary())
#FITTING ON OUR MODEL AND SHOW ELAPSED TIME
start = datetime.datetime.now()
network_history = LSTM_MODEL_SIMCHI.fit(x_train,y_train, epochs=15, batch_size=100,validation_split=0.2)
end = datetime.datetime.now()
elapsed = end - start
print('Total training time : ',str(elapsed))
#EVALUATE OUR MODEL
test_loss,test_acc=LSTM_MODEL_SIMCHI.evaluate(x_test,y_test)
print(test_loss)
print(test_acc)
#PLOT OUR MODEL ON DIAGRAM
history = network_history.history
losses_LSTM_SIMCHI = history['loss']
accuracies_LSTM_SIMCHI = history['acc']
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses_LSTM_SIMCHI)
val_losses = history['val_loss']
plt.plot(val_losses)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(accuracies_LSTM_SIMCHI)
val_accuracies = history['val_acc']
plt.plot(val_accuracies)
plt.legend(['acc','val_acc'])






