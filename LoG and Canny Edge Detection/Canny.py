# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 02:55:45 2021

@author: 302b46
"""

import cv2
import matplotlib.pyplot as plt

# Loading Original Image
image = cv2.imread('../Pictures/dancing_spider.jpeg')

# Applying Canny
edges = cv2.Canny(image, 100, 200, 3, L2gradient=True)

plt.figure()
plt.title('Canny')
plt.imsave('../Pictures/dancing_spider_canny.jpeg', edges, cmap='gray', format='jpeg')
plt.imshow(edges, cmap='gray')
plt.show()