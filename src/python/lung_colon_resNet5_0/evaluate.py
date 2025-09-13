# file: evaluate.py

from tensorflow.keras.models import load_model
from data_loader import get_test_generator

# --- Configuration ---
# NOTE: Update these paths for your environment.
TEST_DATA_DIR = '/content/gdrive/My Drive/CAT_DOGS/2Class/testing_data' # As per the notebook, this seems to be an incorrect path for the lung/colon dataset. Please update it.
SAVED_MODEL_PATH = '/content/gdrive/My Drive/model/best_model.h5'
BATCH_SIZE = 32

# --- Main Evaluation Script ---
if __name__ == "__main__":
    # 1. Load the trained model
    print(f"Loading model from: {SAVED_MODEL_PATH}")
    try:
        model = load_model(SAVED_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained and the path is correct.")
        exit()


    # 2. Load Test Data
    print("Loading test data...")
    try:
        test_generator = get_test_generator(TEST_DATA_DIR, BATCH_SIZE)
    except FileNotFoundError:
        print(f"Error: Test data directory not found at '{TEST_DATA_DIR}'. Please check the path.")
        exit()


    # 3. Evaluate the model
    print("Evaluating model performance on the test set...")
    score = model.evaluate(test_generator)

    print(f"Test Loss: {score[0]}")
    print(f"Test Accuracy: {score[1]}")