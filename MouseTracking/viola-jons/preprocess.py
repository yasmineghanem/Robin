import cv2
import numpy as np
import os
import sys

img = cv2.imread('imgs/img1.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv2.imshow('image', img)    
cv2.waitKey(0)
cv2.destroyAllWindows()