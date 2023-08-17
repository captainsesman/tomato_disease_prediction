# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %%
# Constants
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 256
BATCH_SIZE = 32


# %%
# Extract the Tomato Images from the Zipped File
def load_data(image_directory, img_size, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=image_directory,
    )

    return dataset


# %%
# Preprocessing and Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1)
])

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

# %%
# @tf.autograph.experimental.do_not_convert
# def preprocess_augment(ds, train=True):
#
#     ds = ds.map(lambda x, y: (normalization_layer(x), y))
#     ds = ds.cache()
#     ds = ds.batch(BATCH_SIZE)
#
#     if train:
#         ds = ds.shuffle(1000)
#         ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
#
#     return ds.prefetch(buffer_size=AUTOTUNE)


# %%
def visualize_images(ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in ds:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(class_names[labels[i]])
            plt.axis("off")

