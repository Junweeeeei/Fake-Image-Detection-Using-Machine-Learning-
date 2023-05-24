import cv2
import sys
import time
import numpy as np
from keras import layers
from keras.applications import InceptionResNetV2
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from matplotlib import pyplot

Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

learning_rate=0.0001
batch_size=32
epoch=10
optimizer = Adam(lr = learning_rate , decay = 0.0001/30)

#base_path = "C:/Users/Admin/Desktop/FYP/140k_dataset/" #140k-Real-Fake Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/PS-battles_dataset/" #PS-Battles Dataset
base_path = "C:/Users/Admin/Desktop/FYP/CASIA_dataset/" #CASIA V2.0 Dataset

inception_resnet = InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)  

image_gen = ImageDataGenerator(rescale=1./255.)

train_flow = image_gen.flow_from_directory(
    base_path + 'train/',
    target_size=Image_Size,
    batch_size=batch_size,
    class_mode='binary'
)

image_gen1 = ImageDataGenerator(rescale=1./255.)

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

def build_model(pretrained):
    model = Sequential([
        pretrained,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr = 0.0001 , decay = 0.0001/30),
        metrics=['accuracy',]
    )
    
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

    model = build_model(inception_resnet)
    model.summary()

    history = model.fit_generator(
        train_flow,
        epochs = epoch,
        steps_per_epoch =train_steps,
        validation_data =valid_flow,
        validation_steps = valid_steps
    )

    model.save('InceptionResnet_V2.h5')
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