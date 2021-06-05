# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:16:22 2021

@author: 302b46
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



# Loading original image
image = cv.imread('../Pictures/dancing_spider.jpeg', cv.COLOR_BGR2GRAY)
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Applying gray scale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Roberts Operator
kernelX = np.array([[-1, 0], [0, 1]], dtype=int)
kernelY = np.array([[0, -1],[1, 0]], dtype=int)

# Filtering 
x = cv.filter2D(gray_image, cv.CV_16S, kernelX)
y = cv.filter2D(gray_image, cv.CV_16S, kernelY)

# Turn uint8, image fusion
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
roberts_image = cv.addWeighted(absX, 0.5, absY, 0.5, 0)


plt.figure()
plt.title('Roberts Operator')
plt.imsave('../Pictures/dancing_spider_roberts.jpeg',roberts_image, cmap='gray', format='jpeg')
plt.imshow(roberts_image, cmap='gray')
plt.show()


