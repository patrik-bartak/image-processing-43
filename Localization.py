import cv2
import numpy as np
from PIL import Image

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(image):
    # Replace the below lines with your code.
    # https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python
    # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    result = mask

    #https://subscription.packtpub.com/book/data/9781789344912/7/ch07lvl1sec81/thresholding-color-images
    # (b, g, r) = cv2.split(image)
    # ret2, thresh2 = cv2.threshold(b, 0, 100, cv2.THRESH_BINARY)
    # ret3, thresh3 = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)
    # ret4, thresh4 = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY)
    # result = cv2.merge((thresh2, thresh3, thresh4))


    return result


