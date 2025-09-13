# -*- coding: utf-8 -*-
"""
Evaluates the trained CNN model on the test data.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import data_loader
import config

def evaluate():
    """
    Main evaluation function.
    """
    # Load the best model
    saved_model = load_model(config.MODEL_PATH)

    # Get the test data generator
    test_generator = data_loader.get_test_generator()

    # Evaluate the model
    score = saved_model.evaluate(test_generator)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    # Generate predictions
    Y_pred = saved_model.predict(test_generator, config.NUM_TEST_SAMPLES // config.BATCH_SIZE + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    # Print Confusion Matrix and Classification Report
    print('\nConfusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))

    print('\nClassification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=config.TARGET_NAMES))

if __name__ == '__main__':
    evaluate()