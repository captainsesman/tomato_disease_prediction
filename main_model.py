# %%
import tensorflow as tf

# %%
# Constants
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 11


# %%
# Load Data
def load_data(image_directory, img_size, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=image_directory,
        color_mode='rgb',
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    return dataset


# %%
# Training and Validation Paths
training_path = 'data/archive/train'
val_path = 'data/archive/valid'

# %%
# Constants Definitions
training_ds = load_data(training_path, IMG_SIZE, BATCH_SIZE)
val_ds = load_data(val_path, IMG_SIZE, BATCH_SIZE)
INPUT_SHAPE = training_ds.element_spec[0].shape[1:]


# %%
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


# %%
ds_train = training_ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# %%
# Constants Definitions
print(INPUT_SHAPE)

# %%
ds_test = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(11)
])

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# %%
model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
