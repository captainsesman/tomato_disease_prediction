#%%
from PIL import Image
import os

#%%
def convert_to_jpg(image_path):
    image = Image.open(image_path)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    jpg_path = os.path.splitext(image_path)[0] + ".jpg"
    image.save(jpg_path, "JPEG")

    os.remove(image_path)  # Delete the old image
    print(f"Converted {image_path} to JPG and deleted the original")


def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                continue

            image_path = os.path.join(root, file)
            convert_to_jpg(image_path)

#%%
root_folder = "data/archive"
train_folder = os.path.join(root_folder, "train")
valid_folder = os.path.join(root_folder, "valid")

process_folder(train_folder)
process_folder(valid_folder)
