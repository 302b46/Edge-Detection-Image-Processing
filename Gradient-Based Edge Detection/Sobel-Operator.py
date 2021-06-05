# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:16:22 2021

@author: 302b46
"""



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Loading original image
image = np.array(Image.open('../Pictures/dancing_spider.jpeg')).astype(np.uint8)

# Applying gray scale
gray_image = np.round(0.299 * image[:, :, 0] +
                    0.587 * image[:, :, 1] +
                    0.114 * image[:, :, 2]).astype(np.uint8)

# Sobel Operator
height, width = gray_image.shape

# Defining filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

# Defining images with 0s
newHorizontalImage = np.zeros((height, width))
newVerticalImage = np.zeros((height, width))
newGradientImage = np.zeros((height, width))

# offset by 1
for i in range(1, height - 1):
    for j in range(1, width - 1):
        horizontalGradient = (horizontal[0, 0] * gray_image[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_image[i - 1, j]) + \
                         (horizontal[0, 2] * gray_image[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_image[i, j - 1]) + \
                         (horizontal[1, 1] * gray_image[i, j]) + \
                         (horizontal[1, 2] * gray_image[i, j + 1]) + \
                         (horizontal[2, 0] * gray_image[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_image[i + 1, j]) + \
                         (horizontal[2, 2] * gray_image[i + 1, j + 1])

        newHorizontalImage[i - 1, j - 1] = abs(horizontalGradient)

        verticalGradient = (vertical[0, 0] * gray_image[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_image[i - 1, j]) + \
                       (vertical[0, 2] * gray_image[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_image[i, j - 1]) + \
                       (vertical[1, 1] * gray_image[i, j]) + \
                       (vertical[1, 2] * gray_image[i, j + 1]) + \
                       (vertical[2, 0] * gray_image[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_image[i + 1, j]) + \
                       (vertical[2, 2] * gray_image[i + 1, j + 1])

        newVerticalImage[i - 1, j - 1] = abs(verticalGradient)

        # Edge Value
        edge_value = np.sqrt(pow(horizontalGradient, 2.0) + pow(verticalGradient, 2.0))
        newGradientImage[i - 1, j - 1] = edge_value

plt.figure()
plt.title('Sobel Operator')
plt.imsave('../Pictures/dancing_spider_sobel.jpeg', newGradientImage, cmap='gray', format='jpeg')
plt.imshow(newGradientImage, cmap='gray')
plt.show()
