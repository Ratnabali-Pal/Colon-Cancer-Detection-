# file: data_loader.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators(train_data_dir, batch_size, target_size=(150, 150)):
    """
    Creates and returns the training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # set validation split
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'  # set as training data
    )

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,  # same directory as training data
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'  # set as validation data
    )

    return train_generator, validation_generator

def get_test_generator(test_data_dir, batch_size, target_size=(150, 150)):
    """
    Creates and returns the test data generator.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return test_generator