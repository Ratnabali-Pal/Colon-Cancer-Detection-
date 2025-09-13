# -*- coding: utf-8 -*-
"""
Trains the CNN model for lung and colon cancer classification.
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
import data_loader
import model
import config

def train():
    """
    Main training function.
    """
    # Get data generators
    train_generator, validation_generator = data_loader.get_train_validation_generators()

    # Build the model
    cnn_model = model.build_model()

    # Define callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.PATIENCE)
    mc = ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Train the model
    history = cnn_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.NUM_EPOCHS,
        callbacks=[es, mc]
    )
    print("Training finished.")

if __name__ == '__main__':
    train()