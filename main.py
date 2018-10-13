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
import cv2
train_datagen=ImageDataGenerator(rescale=1./255)#,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

# train_dir="C:/Users/Administrator/Desktop/all/train/train"
# train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=32,class_mode='binary')
# test_dir="C:/Users/Administrator/Desktop/all/train/test"
# test_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')


# mynet=Sequential()

# mynet.add(Conv2D(filters=32,kernel_size=(3, 3),strides=(1,1),input_shape=(150,150,3)))
# mynet.add(Activation('relu'))

# mynet.add(MaxPool2D(pool_size=(2, 2)))
# mynet.add(Conv2D(filters=64,kernel_size=(3, 3),strides=(1,1)))
# mynet.add(Activation('relu'))

# mynet.add(MaxPool2D(pool_size=(2, 2)))
# mynet.add(Conv2D(filters=128,kernel_size=(3, 3),strides=(1,1)))
# mynet.add(Activation('relu'))

# mynet.add(MaxPool2D(pool_size=(2, 2)))
# mynet.add(Conv2D(filters=128,kernel_size=(3, 3),strides=(1,1)))
# mynet.add(Activation('relu'))

# mynet.add(Flatten())
# mynet.add(Dropout(0.5))

# mynet.add(Dense(512))
# mynet.add(Activation('relu'))

# mynet.add(Dense(1))#
# mynet.add(Activation('sigmoid'))
# mynet.summary()
# sgd =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# mynet.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])#categorical
# #plot_model(mynet, to_file='model_size.png',show_shapes=True)


# hist=mynet.fit_generator(train_generator,steps_per_epoch=45,epochs=50)#,validation_data=test_generator,validation_steps=50
# mynet.save("net.h5")
# plt.plot(hist.history['loss'])
# #plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("test_loss.png")
# plt.clf()
# plt.plot(hist.history['acc'])
# #plt.plot(hist.history['val_acc'])
# plt.title('model acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("acc_2.png")

# for data_batch,labels_batch in train_generator:
#     print('data shape:',data_batch.shape)
#     print('labels shape:',labels_batch.shape)
#     break
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list

test_model=load_model("net.h5")
src=cv2.imread("C:/Users/Administrator/Desktop/all/train/redata/dog.258.jpg")
src=cv2.resize(src,(150,150))#
src=src.reshape((1,150,150,3))
src=src.astype("float32")
src=src/255.0
re=test_model.predict(src)
print(re)
q=test_model.predict_classes(src)
print(q)
