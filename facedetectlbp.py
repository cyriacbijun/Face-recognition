#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
#import matplotlib library
import matplotlib.pyplot as plt
import numpy as np
#importing time library for speed comparisons of both classifiers
import time 
get_ipython().run_line_magic('matplotlib', 'inline')
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#mathplotlib requires the above function as it accepts only RGB values while cv.imread accepts BGR values
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
#here I used the lbp classifier
test1 = cv2.imread('test1.jpg')
#here I used a test image test1.jpg,you can replace it with any .jpg file
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
faces = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=7)
print('Faces found: ', len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(convertToRGB(test1))


# In[ ]:




