# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:55:11 2021

@author: 302b46
"""
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../Pictures/dancing_spider.jpeg')

#Applying Gray Scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Applying Gaussian Blur
gaussian_blur_image = cv2.GaussianBlur(gray_image, (3,3), 0)


#Laplacian Operator 

laplacian_image = cv2.Laplacian(gaussian_blur_image,cv2.CV_64F)


plt.figure()
plt.title('Laplacian')
plt.imsave('../Pictures/laplacian-dancing_spider.jpeg', laplacian_image, cmap='gray', format='jpeg')
plt.imshow(laplacian_image, cmap='gray')
plt.show()
