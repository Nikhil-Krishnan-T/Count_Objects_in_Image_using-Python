import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

image = cv2.imread("D:/NKT_GIT/Count-Object-In-Image/cv2/cars.jpg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(image)
plt.show()
print("Number of cars in this image are " +str(label.count('car')))