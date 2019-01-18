import cv2
#import matplotlib library
import matplotlib.pyplot as plt
import numpy as np
#importing time library for speed comparisons of both classifiers
import time 

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#mathplotlib requires the above function as it accepts only RGB values while cv.imread accepts BGR values
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#here I used the haar classifier
test1 = cv2.imread('test1.jpg')
#here I used a test image test1.jpg,you can replace it with any .jpg file
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print('Faces found: ', len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(convertToRGB(test1))






