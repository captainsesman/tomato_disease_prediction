from Data_Preparation.data_preparation import  load_data
from Models.sequential_api import model
import tensorflow as tf

# %%
# Constants
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 11

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
my_model = model(NUM_CLASSES, input_shape=[IMG_SIZE, IMG_SIZE, 3])

# %%
small_train = training_ds.take(1000)
small_val_ds = val_ds.take(1000)

#%%
my_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# %%
history = my_model.fit(
    small_train,
    validation_data=small_val_ds,
    epochs=20,
)

#%%
import matplotlib.pyplot as plt
training_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 10))
plt.plot(val_loss,)
plt.plot(training_loss)
plt.title("Validation and Training Loss ")
plt.show()

#%%

my_model.save("tomato.hd5")