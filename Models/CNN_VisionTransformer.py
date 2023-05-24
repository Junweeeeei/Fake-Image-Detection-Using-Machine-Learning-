
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageChops, ImageEnhance
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import InceptionResNetV2, ResNet50
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input, Reshape
from sklearn import metrics
from keras.optimizers import SGD, Adam, RMSprop
from vit_keras import vit


image_size=224
image_channels = 3
input_shape = (image_size  , image_size  , image_channels)
learning_rate = 0.0006
weight_decay = 0.0001
num_epochs = 10
batch_size = 128

# specify the path to your data folder
#base_path = "C:/Users/Admin/Desktop/FYP/140k_dataset/" #140k-Real-Fake Dataset
base_path = "C:/Users/Admin/Desktop/FYP/PS-battles_dataset/" #PS-Battles Dataset
#base_path = "C:/Users/Admin/Desktop/FYP/CASIA_dataset/" #CASIA V2.0 Dataset

# set up the ImageDataGenerator with appropriate data augmentation
datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=None)


# use the datagen.flow_from_directory() method to load the images
train_flow = datagen.flow_from_directory(
    base_path + 'train/',
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='binary'
)

valid_flow = datagen.flow_from_directory(
    base_path + 'val/', 
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_flow = datagen.flow_from_directory(
    base_path + 'test/',
    target_size=(image_size,image_size),
    batch_size=1,   
    shuffle = False, # Prevents the images from being shuffled
    class_mode='binary'
)

config = {}
config["num_layers"] = 8
config["hidden_dim"] = 128
config["mlp_dim"] = 1028
config["num_heads"] = 4
config["dropout_rate"] = 0.1

config["image_size"] = 224
config["patch_size"] = 16
config["num_patches"] = 196
config["num_channels"] = 3
config["num_classes"] = 1

class ClassToken(layers.Layer):
    def __init__(self,):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype="float32"),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)

        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = layers.Add()([x, skip_1])

    skip_2 = x
    x = layers.LayerNormalization()(x)
    x = mlp(x, cf)
    x = layers.Add()([x, skip_2])

    return x

def ResNet50ViT(cf):
    """ Input """
    inputs = Input((cf["image_size"], cf["image_size"], cf["num_channels"])) ## (None, 256, 256, 3)
    x = tf.keras.layers.Conv2D(16, kernel_size=3, strides= 2,padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32,kernel_size=3, strides= 2,padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=3, strides= 2,padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256,kernel_size=3, strides= 2,padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=1, strides= 1, activation='relu')(x)
    '''x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)     '''
    
    output = x
    
    """ Pre-trained Resnet50 
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    output = resnet50.output ## (None, 7, 7, 2048) """

    """ Patch Embeddings """
    patch_embed = Conv2D(
        cf["hidden_dim"],
        kernel_size=cf["patch_size"],
        padding="same"
    )(output)       ## (None, 14,14,128)
    _, h, w, f = patch_embed.shape
    patch_embed = Reshape((h*w, f))(patch_embed) ## (None, 196, 128)


    """ Position Embeddings """
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1) ## (196,)
    pos_embed = layers.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)

    """ Patch + Position Embeddings """
    embed = patch_embed + pos_embed ## (None, 196, 128)

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = layers.Concatenate(axis=1)([token, embed]) ## (None, 197, 128)

    """ Transformer Encoder """
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    x = layers.LayerNormalization()(x)
    x = x[:, 0, :]
    x = Dense(cf["num_classes"], activation="softmax")(x)

    model = Model(inputs, x)
    return model

def run_experiment():
    # start a timer
    start=time.time()
    starttime=time.ctime(start)
    print("Timer Start at:" , starttime)

    model = ResNet50ViT(config)

    model.summary() 

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    decay_steps = len(train_flow) # batch size
    initial_learning_rate = learning_rate

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                    min_delta = 1e-4,
                                                    patience = 5,
                                                    mode = 'max',
                                                    restore_best_weights = True,
                                                    verbose = 1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = './IdiotDev_Checkpoints.hdf5',
                                                    monitor = 'val_accuracy', 
                                                    verbose = 1, 
                                                    save_best_only = True,
                                                    save_weights_only = True,
                                                    mode = 'max')
    
    callbacks = [earlystopping, lr_scheduler, checkpointer]

    train_steps = len(train_flow) # Step size which is also the batch size
    valid_steps = len(valid_flow)

    history = model.fit(train_flow,
          steps_per_epoch = train_steps,
          validation_data = valid_flow,
          validation_steps = valid_steps,
          epochs = num_epochs,
          callbacks = callbacks)

    y_pred = model.predict(test_flow)
    y_test = test_flow.classes

    print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred > 0.5, digits=4))
    
    # end timer
    end=time.time()
    endtime=time.ctime(end)
    print("Batch Size :", batch_size)
    print("Epochs :", num_epochs)
    print("Learning Rate :", learning_rate)
    print("Optimizer :", optimizer)
    print("Dataset Base Path :", base_path)
    print("Timer End at:" , endtime)
    print("Total Elasped Time In Seconds:", end-start)

    return history


history = run_experiment()