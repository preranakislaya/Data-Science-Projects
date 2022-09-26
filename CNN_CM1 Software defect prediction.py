# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:10:13 2022

@author: Dipu
@project: CNN for classification
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report

# Reading the data
df = pd.read_csv('D:\python_coding\TSL\dataset\promise_data\CM1 Software defect prediction.csv', skiprows = (1,2))
df.head(5)

# Relabelling column names
df.columns = df.columns.str.replace('attribute ', 'a_')
df.columns = df.columns.str.replace(' numeric', '_n')
df.head()

# Separating DV and IV
x = df.drop('a_defects {false,true}', axis = 1)
y = df['a_defects {false,true}']
x.shape

# Splitting train-test data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 17)


# Create the model (addition of layers)
model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(x_train.shape[1],1)))
# a = (x_train.shape[1],1)
# print(a)
model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()

# Fit the model
model.fit(x_train, y_train, validation_split = 0.1, epochs = 70, batch_size = 16, verbose=1)

# Predicting and accuracy check
acc = model.evaluate(x_test,y_test)
print('Accuracy: %.2f%%' % (acc[1]*100))

acc_train = model.evaluate(x_train,y_train) # accuracy of train data

# Prediction of the test data
pred = model.predict(x_test)
y_pred = pred.argmax(axis=-1)

# Prediction accuracy with the confusion matrix
labels = ['False', 'True']
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=labels, columns=labels) # printing confusion matrix
