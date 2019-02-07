# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Image Recognition"""

__author__ = 'Qiyun Lu'


# Section One: build the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def build_classifier(convolution_units, image_size, dense_units, dropout_rate):

    # initialize the CNN
    classifier = Sequential()

    # convolution
    classifier.add(Convolution2D(convolution_units, (3, 3), input_shape=(image_size, image_size, 3), activation='relu', kernel_initializer='he_normal'))
    # pooling
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    # second convolution layer and pooling layer
    classifier.add(Convolution2D(convolution_units * 2, (3, 3), activation='relu', kernel_initializer='he_normal'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    # third convolution layer and pooling layer
    classifier.add(Convolution2D(convolution_units * 2, (3, 3), activation='relu', kernel_initializer='he_normal'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    # flattening
    classifier.add(Flatten())

    # full connection
    classifier.add(Dense(dense_units, activation='relu'))
    classifier.add(Dropout(dropout_rate))
    classifier.add(Dense(dense_units * 2, activation='relu'))
    classifier.add(Dropout(dropout_rate * 2))
    classifier.add(Dense(1, activation='sigmoid'))

    # compile the CNN
    optimizer = Adam()
    classifier.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

# build the classifier
convolution_units = 32
image_size = 150
dense_units = 64
dropout_rate = 0.2
classifier = build_classifier(convolution_units, image_size, dense_units, dropout_rate)


# Section Two: fit the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
batch_size = 128
epochs = 90

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary')
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000//batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=2000//batch_size,
    max_queue_size=100,
    workers=12)


# Section Three: make a single prediction

import numpy as np
from keras.preprocessing import image

test_image_name = 'cat_or_dog_1.jpg'
test_image = image.load_img('dataset/single_prediction/'+test_image_name, target_size=(image_size, image_size))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
if result[0][0] == training_set.class_indices['dogs']:
    prediction = 'dog'
else:
    prediction = 'cat'
print("Image", test_image_name, "is", prediction)

