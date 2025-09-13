# file: model.py

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def create_resnet50_model(input_shape=(150, 150, 3), learning_rate=0.001):
    """
    Creates, compiles, and returns a ResNet50 model for binary classification.
    """
    # Load ResNet50 model without classifier layers
    base_model = ResNet50(include_top=False, input_shape=input_shape)

    # Add new classifier layers
    flat1 = Flatten()(base_model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(2, activation='softmax')(class1)

    # Define new model
    model = Model(inputs=base_model.inputs, outputs=output)

    # Compile the model
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model