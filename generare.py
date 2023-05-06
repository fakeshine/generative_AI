# -*- coding: utf-8 -*-
"""generare.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iDBoqBjlb-5glYe2lxKp3HBjksO8Ffb0
"""

# Model
from tensorflow import keras
import  keras_cv
keras.mixed_precision.set_global_policy("mixed_float16")

# Visualization
import matplotlib.pyplot as plt

# Save the image
from PIL import Image

"""### Build Model"""

# Create a model
model = keras_cv.models.StableDiffusion(img_height=512, 
                                        img_width=512,
                                        jit_compile=True)

def plot_images(images):
    # Set figure size
    plt.figure(figsize=(20, 20))
    # Loop through each image
    for i in range(len(images)):
        # Subplot setup
        ax = plt.subplot(1, len(images), i + 1)
        # Plot each image
        plt.imshow(images[i])
        # Do not show axis
        plt.axis("off")

# Create images from text
images = model.text_to_image(prompt="A realistic image of a navy-colored zebra",
                             batch_size=5)

# Plot the images
plot_images(images)

# Create images from text
images_2nd_try = model.text_to_image(prompt="Generate an image of a navy-colored zebra. The background should be minimalistic and any color except for navy and white. Use a realistic style with shading and texture",
                             batch_size=3)

# Plot the images
plot_images(images_2nd_try)

# Commented out IPython magic to ensure Python compatibility.
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change directory
import os
if not os.getcwd() == '/content/drive/MyDrive/navy_zebra':
#     %cd /content/drive/MyDrive/navy_zebra

# Save images

for i, image in enumerate(images):
  filename = f"navy_zebra{i+1}.jpeg"
  if not os.path.isfile(filename):
    Image.fromarray(image).save(filename)
  else:
    print(filename + " already exists")

for i, image in enumerate(images_2nd_try):
  filename = f"navy_zebra_2nd_try_{i+1}.jpeg"
  if not os.path.isfile(filename):
    Image.fromarray(image).save(filename)
  else:
    print(filename + " already exists")

!pwd