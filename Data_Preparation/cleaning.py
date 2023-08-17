#%%
import os
import tensorflow as tf

#%%
def is_valid_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        _ = tf.image.decode_image(image)
        return True
    except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
        return False
def load_data(data_dir):
    # Initialize a list to store valid file paths
    valid_files = []

    # Identify and delete corrupted or unsupported files
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_image(file_path):
                continue
            else:
                print(f"Deleting file: {file_path} due to unsupported format or corruption.")
                os.remove(file_path)
#%%
# Usage
data_dir = 'data/archive/train'
img_size = 256
batch_size = 32
training_ds = load_data(data_dir)
