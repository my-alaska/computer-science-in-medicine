# AGH UST Medical Informatics 03.2021
# Lab 2 : Segmentation

import cv2 as cv
import numpy as np

im = cv.imread("data/abdomen.png")
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
w = im.shape[1]
h = im.shape[0]

mask = np.zeros([h, w], np.uint8)

def mouse_callback(event, x, y, flags, params):
    if event == 1:
        print([x, y])
        print(im[y, x])


cv.imshow("image", im)
cv.setMouseCallback("image", mouse_callback)
cv.waitKey()
cv.destroyAllWindows()
