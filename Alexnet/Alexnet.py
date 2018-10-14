import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from  keras.utils import np_utils
from keras import metrics
import keras
from keras.models import save_model,load_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.utils import plot_model
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image
from keras.preprocessing import image

conv_base=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))#
train_dir="C:/Users/Administrator/Desktop/all/train/train"
test_dir="C:/Users/Administrator/Desktop/all/train/test"


datagen=ImageDataGenerator(1./255)
batch_size=32

def extract_feature(directory,sample_count):
    features=np.zeros(shape=(sample_count,1,1,2048))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(224,224),batch_size=batch_size,class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

# save data
# train_feature,train_labels=extract_feature(train_dir,1420)
# np.save(open('Alexnet/train_feature_Alexnet.npy','wb+'),train_feature)
# np.save(open('Alexnet/train_labels_Alexnet.npy','wb+'),train_labels)
# print('train save!')
# test_feature,test_labels=extract_feature(test_dir,700)
# np.save(open('Alexnet/test_feature_Alexnet.npy','wb+'),test_feature)
# np.save(open('Alexnet/test_labels_Alexnet.npy','wb+'),test_labels)
# print('test save!')

train_feature=np.load(open('Alexnet/train_feature_Alexnet.npy','rb'))
train_labels=np.load(open('Alexnet/train_labels_Alexnet.npy','rb'))
test_feature=np.load(open('Alexnet/test_feature_Alexnet.npy','rb'))
test_labels=np.load(open('Alexnet/test_labels_Alexnet.npy','rb'))
train_feature=np.reshape(train_feature,(1420,1*1*2048))
test_feature=np.reshape(test_feature,(700,1*1*2048))

mynet=Sequential()
mynet.add(Dense(1024,input_dim=1*1*2048))
mynet.add(Activation('relu'))
mynet.add(Dropout(0.5))

mynet.add(Dense(1))#
mynet.add(Activation('sigmoid'))

sgd =SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
mynet.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])#categorical

hist=mynet.fit(train_feature,train_labels,epochs=40,batch_size=16,validation_data=(test_feature,test_labels))#
mynet.save("Alexnet/Alexnet.h5")
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("Alexnet/Alexnet_loss40.png")
plt.clf()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("Alexnet/Alexnet40.png")