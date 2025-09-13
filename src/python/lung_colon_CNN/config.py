# -*- coding: utf-8 -*-
"""
Configuration file for the lung and colon cancer classification project.
"""

# Path to the training and validation data directory
TRAIN_DATA_DIR = '/content/gdrive/My Drive/lung_colon_image_set/colon_image_sets'

# Path to the testing data directory
TEST_DATA_DIR = '/content/gdrive/My Drive/CAT_DOGS/2Class/testing_data'

# Path to save the best model
MODEL_PATH = '/content/gdrive/My Drive/model/best_model.h5'

# Image dimensions
IMG_WIDTH = 150
IMG_HEIGHT = 150

# Model training parameters
BATCH_SIZE = 45
NUM_EPOCHS = 1500
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Early stopping patience
PATIENCE = 50

# Number of test samples
NUM_TEST_SAMPLES = 400

# Class names
TARGET_NAMES = ['adenocarcinomas', 'benign']