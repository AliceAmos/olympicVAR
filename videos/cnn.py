from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
# from keras.utils import to_categorical
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()
# print(X_train.shape, y_train.shape)

def load_data():
    annots = loadmat('videos/split_4_train_list.mat')
    arranged = [[element for element in upperElement] for upperElement in annots['consolidated_train_list']]
    new_data = list()
    for i in arranged:
        new_data.append((i[0], i[1], i[2]))
    columns = ['class', 'video no.', 'score']
    data = pd.DataFrame(new_data, columns=columns)
    data = data.sort_values(["class", "video no."])

    return data

def try_train():
    x_train = np.ndarray((0,240,320,3))
    for i in range(1,9):
        path = "diving/00" + str(i)+".avi"
        print(path)
        cap = cv2.VideoCapture(path)
        frames = []
        ret = True
        while ret:
            ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(img)
        # video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
        x_train = np.concatenate((x_train,frames))
    return x_train

def try_test():
    x_test = np.ndarray((0,240,320,3))
    for i in range(0,7):
        path = "diving/01" + str(i)+".avi"
        print(path)
        cap = cv2.VideoCapture(path)
        frames = []
        ret = True
        while ret:
            ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(img)
        x_test = np.concatenate((x_test,frames))
    return x_test
    
x_train = try_train()
x_train =  x_train.reshape(8, 103,  240, 320, 3)
y_train = np.array((76.5, 72, 83.23, 76.5 , 80, 86.4 , 81.6, 97.2))
print(x_train.shape, y_train.shape)
input_shape = (103,240,320,3)
x_test = try_test()
x_test =  x_test.reshape(7, 103,  240, 320, 3)
y_test = np.array((97.2, 91.2, 91.2, 86.4, 88.8, 82.5, 88.2))



#reshaping data
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
# X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)) 
# #checking the shape after reshaping
# print(X_train.shape)
# print(X_test.shape)
# #normalizing the pixel values
# X_train=X_train/255
# X_test=X_test/255

#defining model
# model=Sequential()
# #adding convolution layer
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(240,320,3)))
# #adding pooling layer
# model.add(MaxPool2D(2,2))
# #adding fully connected layer
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# #adding output layer
# model.add(Dense(10,activation='softmax'))
# #compiling the model
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.add(Dense(10, activation='softmax'))
#fitting the model
model.fit(x_train, y_train,epochs=10)
model.evaluate(x_test, y_test)