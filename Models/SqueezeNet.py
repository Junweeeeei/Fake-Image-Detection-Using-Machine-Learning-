import tensorflow as tf

import sys
import time
import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns
from numpy import load
from matplotlib import pyplot

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Concatenate, Add, Input
from keras.layers import Dense,GlobalAveragePooling2D,Conv2D,BatchNormalization, Activation
from keras.layers import Flatten,MaxPooling2D,Dropout
import keras.backend as K

from keras.applications.densenet import preprocess_input

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, Sequential

from keras.optimizers import Adam



Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

learning_rate=0.00001
batch_size=32
epoch=10
optimizer = Adam(lr = learning_rate , decay = 0.0001/30)

def SqueezeNet(input_shape, nb_classes, use_bypass=False, dropout_rate=None, compression=1.0):
    """
    Creating a SqueezeNet of version 1.1
    
    2.4x less computation over SqueezeNet 1.0 implemented above.
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps
        
    Returns:
        Model        : Keras model instance
    """
    
    input_img = Input(shape=input_shape)

    x = Conv2D(int(64*compression), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2')
    x = create_fire_module(x, int(16*compression), name='fire3')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool3')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire4')
    x = create_fire_module(x, int(32*compression), name='fire5')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool5')(x)
    
    x = create_fire_module(x, int(48*compression), name='fire6')
    x = create_fire_module(x, int(48*compression), name='fire7')
    x = create_fire_module(x, int(64*compression), name='fire8')
    x = create_fire_module(x, int(64*compression), name='fire9')

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)

def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret

def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis

def output(x, nb_classes):
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x


squeezenet_model = SqueezeNet( nb_classes=2,input_shape=(Image_Width,Image_Height,3))


base_path = "C:/Users/Admin/Desktop/FYP/140k_dataset/" #140k-Real-Fake Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/PS-battles_dataset/" #PS-Battles Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/CASIA_dataset/" #CASIA V2.0 Dataset

# prepare data generators
image_gen = ImageDataGenerator(rescale=1./255.)
image_gen1 = ImageDataGenerator(rescale=1./255.)

train_flow = image_gen.flow_from_directory(
    base_path + 'train/',
    target_size=Image_Size,
    batch_size=batch_size,
    class_mode='binary'
)

valid_flow = image_gen1.flow_from_directory(
    base_path + 'val/',
    target_size=Image_Size,
    batch_size=batch_size,
    class_mode='binary'
)

test_flow = image_gen.flow_from_directory(
    base_path + 'test/',
    target_size=Image_Size,
    batch_size=1,
    shuffle = False,
    class_mode='binary'
)

# define cnn model
def define_model():
	model = Sequential()
	model.add(squeezenet_model)
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0001, decay=0.0001 / 30)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summary_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='validation')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='validation')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
    # start a timer
    start=time.time()
    starttime=time.ctime(start)
    print("Timer Start at:" , starttime)   

    train_steps = len(train_flow)
    valid_steps = len(valid_flow)

    model = define_model()
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                            min_delta = 0,
                            patience = 2,
                            verbose = 0,
                            mode = 'auto')

    history = model.fit_generator(
        train_flow,
        epochs = epoch,
        steps_per_epoch =train_steps,
        validation_data =valid_flow,
        validation_steps = valid_steps,
        callbacks = [early_stopping]
    )

    model.save('SqueezeNet.h5') #saving model...

    # evaluate model
    # learning curves
    summary_diagnostics(history)

    y_pred = model.predict(test_flow)
    y_test = test_flow.classes

    print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred > 0.5, digits=4))

    # end timer
    end=time.time()
    endtime=time.ctime(end)
    print("Batch Size :", batch_size)
    print("Epochs :", epoch)
    print("Learning Rate :", learning_rate)
    print("Optimizer :", optimizer)
    print("Dataset Base Path :", base_path)
    print("Timer End at:" , endtime)
    print("Total Elasped Time In Seconds:", end-start)
 
# entry point, run the test harness
run_test_harness()