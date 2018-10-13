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
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import cv2
from PIL import Image
from keras.preprocessing import image
#train_datagen=ImageDataGenerator(rescale=1./255)#,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#test_datagen=ImageDataGenerator(rescale=1./255)

train_dir="C:/Users/Administrator/Desktop/all/train/train"
#train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=32,class_mode='binary')
test_dir="C:/Users/Administrator/Desktop/all/train/test"
#test_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=32,class_mode='binary')


conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
conv_base.summary()
datagen=ImageDataGenerator(1./255)
batch_size=32

def extract_feature(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

# train_feature,train_labels=extract_feature(train_dir,1420)
# np.save(open('train_feature.npy','wb+'),train_feature)
# np.save(open('train_labels.npy','wb+'),train_labels)
# print('train save!')
# test_feature,test_labels=extract_feature(test_dir,700)
# np.save(open('test_feature.npy','wb+'),test_feature)
# np.save(open('test_labels.npy','wb+'),test_labels)
# print('test save!')

# train_feature=np.load(open('train_feature.npy','rb'))
# train_labels=np.load(open('train_labels.npy','rb'))
# test_feature=np.load(open('test_feature.npy','rb'))
# test_labels=np.load(open('test_labels.npy','rb'))
# train_feature=np.reshape(train_feature,(1420,4*4*512))
# test_feature=np.reshape(test_feature,(700,4*4*512))

# mynet=Sequential()
# mynet.add(Dense(256,input_dim=4*4*512))
# mynet.add(Activation('relu'))
# mynet.add(Dropout(0.5))

# mynet.add(Dense(1))#
# mynet.add(Activation('sigmoid'))
# mynet.summary()
# sgd =SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# mynet.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])#categorical
# #plot_model(mynet, to_file='VGG_model_size.png',show_shapes=True)


# hist=mynet.fit(train_feature,train_labels,epochs=30,batch_size=16,validation_data=(test_feature,test_labels))#
# mynet.save("vgg.h5")
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("VGG_loss_1.png")
# plt.clf()
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("acc_VGG_1.png")


