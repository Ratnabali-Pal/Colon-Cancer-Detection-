# -*- coding: utf-8 -*-
"""
Defines the CNN model architecture using VGG16 as a base.
"""

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
import config

def build_model():
    """
    Builds and compiles the VGG16-based model.
    """
    # Load VGG16 model without classifier layers
    base_model = VGG16(include_top=False, input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))

    # Add new classifier layers
    flat1 = Flatten()(base_model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(2, activation='softmax')(class1)

    # Define new model
    model = Model(inputs=base_model.inputs, outputs=output)

    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model