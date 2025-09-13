# file: train.py

import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import get_train_val_generators
from model import create_resnet50_model

# --- Configuration ---
# NOTE: For a local setup, you would change this path.
# For Google Colab, you would first mount your drive:
# from google.colab import drive
# drive.mount('/content/gdrive')
TRAIN_DATA_DIR = '/content/gdrive/My Drive/lung_colon_image_set/colon_image_sets'
MODEL_SAVE_PATH = '/content/gdrive/My Drive/model/best_model.h5'

# Training parameters
BATCH_SIZE = 45
NB_EPOCHS = 1000
PATIENCE = 50

# --- Main Training Script ---
if __name__ == "__main__":
    # Create directory to save model if it doesn't exist
    model_save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Created directory: {model_save_dir}")

    # 1. Load Data
    print("Loading data...")
    train_generator, validation_generator = get_train_val_generators(TRAIN_DATA_DIR, BATCH_SIZE)

    # 2. Create Model
    print("Creating model...")
    model = create_resnet50_model()
    model.summary()

    # 3. Define Callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    mc = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=NB_EPOCHS,
        callbacks=[es, mc],
        workers=10
    )
    print("Training finished.")