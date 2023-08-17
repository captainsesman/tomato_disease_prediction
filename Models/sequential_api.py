# %%
from tensorflow import keras
from tensorflow.keras import layers
from Data_Preparation.data_preparation import data_augmentation, resize_and_rescale


# %%
def model(num_classes, input_shape):
    my_model = keras.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    return my_model
