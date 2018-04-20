from keras.models import Sequential
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
import json
import h5py
import os
import _pickle as cPickle
import sys
import shutil
from keras import backend as K
K.set_image_dim_ordering('tf')

from CNN import New_model as CNN

folder = './Data'
train = folder + '/Train'
val = folder + '/Validate'

nb_epoch = 50
img_height, img_width = 256,256

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,    
        rescale=1./255,
        fill_mode='nearest',
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='constant',
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(img_width, img_height),
        batch_size=32  , color_mode="grayscale"
        )

val_generator = val_datagen.flow_from_directory(
        val,
        target_size=(img_width, img_height),
        batch_size=32  , color_mode="grayscale"
        )

model, model_name = CNN(train_generator.num_classes)

model_dir = folder + '/' + model_name + '/'
os.mkdir(model_dir)
temp_path = model_dir + 'CNN-Inter.h5'

checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)


output = model.fit_generator(
                                train_generator,
                                samples_per_epoch=train_generator.samples,
                                nb_epoch=nb_epoch,
                                validation_data=val_generator,
                                nb_val_samples=val_generator.samples,
                                callbacks=[checkpointer])

model.save(model_dir + 'CNN.h5')

model.save_weights("Weights.h5")
print("Saved model to disk")    


