import sys
import time
import random
import os
import numpy as np
from numpy import load
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD , Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn import metrics

Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

learning_rate = 0.0001
batch_size=32
epoch=10
optimizer = Adam(lr = learning_rate , decay = 0.0001/30)

base_path = "C:/Users/Admin/Desktop/FYP/140k_dataset/" #140k-Real-Fake Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/PS-battles_dataset/" #PS-Battles Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/CASIA_dataset/" #CASIA V2.0 Dataset

# create data generators
image_gen = ImageDataGenerator(rescale=1./255.)
image_gen1 = ImageDataGenerator(rescale=1./255.)

# prepare iterators
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
    target_size=(224, 224),
    batch_size=1,
    shuffle = False,
    class_mode='binary'
)

# define cnn model
def define_model():
	'''model = Sequential()
	model.add(Conv2D(60, 3, padding='same', activation='relu', input_shape=(Image_Width,Image_Height,Image_Channels)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Conv2D(120, 3, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Conv2D(200, 3, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Conv2D(200, 3, padding='same', activation='relu'))
	model.add(Conv2D(100, 3, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(1, activation='sigmoid'))'''
	'''model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', activation ='relu', input_shape = (Image_Height,Image_Width,3)))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', activation ='relu'))
	model.add(MaxPooling2D((2, 2))) #No specificity on strides so default strides is same size as pooling which is 2x2 , basically reducing our output of per convulutional layer by 2
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid'))'''
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'valid', activation ='relu', input_shape = (Image_Height,Image_Width,3)))
	model.add(MaxPooling2D((2, 2))) #No specificity on strides so default strides is same size as pooling which is 2x2 , basically reducing our output of per convulutional layer by 2
	model.add(Dropout(0.25))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'valid', activation ='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid'))
	# compile model
	opt = optimizer
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

	# define model
	model = define_model()
	model.summary()
	
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
	# This callback will stop the training when there is no improvement in
	# the loss for three consecutive epochs.

	train_steps = len(train_flow)
	valid_steps = len(valid_flow)

	# fit model
	history = model.fit_generator(
		train_flow,
		epochs = epoch,
		steps_per_epoch =train_steps,
		validation_data =valid_flow,
		validation_steps = valid_steps,
        callbacks = [early_stopping]
	)

	# evaluate model
	model.save('BaselineModel_flow.h5') #saving model...
	_, acc = model.evaluate_generator(valid_flow, steps=len(valid_flow), verbose=0)
	print('> %.3f' % (acc * 100.0))
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


