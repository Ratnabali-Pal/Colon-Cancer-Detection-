# -*- coding: utf-8 -*-
"""
Data loader and preprocessing for the lung and colon cancer dataset.
"""

from keras.preprocessing.image import ImageDataGenerator
import config

def get_train_validation_generators():
    """
    Creates and returns the training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=config.VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DATA_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        config.TRAIN_DATA_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def get_test_generator():
    """
    Creates and returns the test data generator.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DATA_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary'
    )

    return test_generator