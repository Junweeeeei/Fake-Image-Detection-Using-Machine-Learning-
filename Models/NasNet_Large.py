import sys
import time
import random
import os
import numpy as np
from numpy import load
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from keras.applications.resnet import preprocess_input
from sklearn import metrics
from keras.applications.nasnet import NASNetLarge

Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

learning_rate = 0.0001
batch_size=32
epoch=10
optimizer = Adam(lr = learning_rate , decay = 0.0001/30)

nasnet_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(Image_Width,Image_Height,3))
nasnet_model.trainable = False

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
	model.add(nasnet_model)
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
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
		epochs = 10,
		steps_per_epoch =train_steps,
		validation_data =valid_flow,
		validation_steps = valid_steps,
        callbacks = [early_stopping]
	)
	
	model.save('NasNet_Large.h5') #saving model...

	# evaluate model
	# learning curves
	summary_diagnostics(history)

	y_pred = model.predict(test_flow)
	y_test = test_flow.classes

	print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
	print("AP Score:", metrics.average_precision_score(y_test, y_pred))
	print()
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
