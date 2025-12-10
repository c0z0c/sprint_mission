

import numpy as np
import cv2

image = cv2.imread('input.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28,28)).astype(np.float32)/255
input = np.reshape(gray, (1,1,28,28)