#!/usr/bin/env python
# coding: utf-8

# ### Crop Objects


# IMPORT

import cv2
import numpy as np
import os


zebra_images_path = 'navy_zebra_images/'
labels_path = 'runs/detect/exp/labels/'

x_center = []
y_center = []
width = []
height = []

row_top_bbox = []
row_bottom_bbox = []
col_left_bbox = []
col_right_bbox = []

i = 0
# 512x512 = image size

for file in os.listdir(labels_path):

    label_file_path = os.path.join(labels_path, file)
    with open(label_file_path, 'r') as my_file:
        line = my_file.readline()
        parameters = line.split(' ')
        x_center.append(float(parameters[1]) * 512)
        y_center.append(float(parameters[2]) * 512)
        width.append(float(parameters[3]) * 512)
        height.append(float(parameters[4]) * 512)

    row_top_bbox.append(int(y_center[i] - height[i] / 2))
    row_bottom_bbox.append(int(y_center[i] + height[i] / 2))
    col_left_bbox.append(int(x_center[i] - width[i] / 2))
    col_right_bbox.append(int(x_center[i] + width[i] / 2))
    i += 1

# display(row_top_bbox)
# display(row_bottom_bbox)
# display(col_left_bbox)
# display(col_right_bbox)
        
i = 0

prefix_original_images = 'navy'
prefix_cropped_images = 'cropped'

for file in os.listdir(zebra_images_path):
    if file.startswith(prefix_original_images):
        image_path = os.path.join(zebra_images_path, file)
        image = cv2.imread(image_path)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image = image[row_top_bbox[i]:row_bottom_bbox[i], col_left_bbox[i]:col_right_bbox[i]]
        i += 1
        # cv2.imshow(f'cropped_image{i+1}', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_path = 'navy_zebra_images/cropped_' + file
        if not os.path.exists(save_path):
            cv2.imwrite(save_path , image)
        else:
            print('Cropped image already exists')
    



### Grayscale


for file in os.listdir(zebra_images_path):
    if file.startswith(prefix_cropped_images):
        image_path = os.path.join(zebra_images_path, file)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_path = 'navy_zebra_images/gray_' + file
        if not os.path.exists(save_path):
            cv2.imwrite(save_path , gray_image)
        else:
            print('Grayscale image already exists')



# ### Masca Color

height.clear()
width.clear()
dim = []

i = 0

# Added the shapes of the images because when I converted the original height and width arrays to integer values there were slightly different than the actual size of the cropped imaegs

for file in os.listdir(zebra_images_path):
    if file.startswith(prefix_cropped_images):
        image_path = os.path.join(zebra_images_path, file)
        image = cv2.imread(image_path)
        height.append(image.shape[0])
        width.append(image.shape[1])
        dim.append(image.shape[2])
        i = i + 1

lower_navy= np.array([0,0,0])
upper_navy = np.array([128,32,255])
bg_color = np.array([255,255,255])

for file in os.listdir(zebra_images_path):
    if file.startswith(prefix_cropped_images):
        image_path = os.path.join(zebra_images_path, file)
        image = cv2.imread(image_path)
        mask = cv2.inRange(image, lower_navy, upper_navy)
        # masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = image
        masked_image[mask != 255] = bg_color
        save_path = 'navy_zebra_images/masked_' + file
        save_path_mask = 'navy_zebra_images/mask_' + file
        if not os.path.exists(save_path):
            cv2.imwrite(save_path , masked_image)
        else:
            print('Masked image already exists')
        if not os.path.exists(save_path_mask):
            cv2.imwrite(save_path_mask, mask)
        else:
            print('Mask already exists')

        



