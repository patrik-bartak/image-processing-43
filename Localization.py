import cv2
import numpy as np
import imutils

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
    # https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv

    original = image.copy()
    mask_yellow = get_yellow_mask(image)
    mask = mask_yellow

    mask_loc = get_plates_by_bounding(image)

    mask = cv2.bitwise_or(mask, mask_loc)

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #
    # result = cv2.bitwise_and(original, original, mask=mask)

    result = [cv2.bitwise_and(original, original, mask = mask)]

    return result


def get_yellow_mask(image):
    image_yellow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([8, 93, 100], dtype="uint8")
    upper_yellow = np.array([45, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(image_yellow, lower_yellow, upper_yellow)
    return mask_yellow


def get_plates_by_bounding(image):
    #https://www.youtube.com/watch?v=UgGLo_QRHJ8
    all_plates = []
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    image_blur = cv2.bilateralFilter(image_grey, 17, 15, 15)
    image_edges = cv2.Canny(image_blur, 30, 200)
    contours = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours_reduced = sorted(contours, key=cv2.contourArea, reverse = True)[:5]

    for con in contours_reduced:
        perimeter = cv2.arcLength(con, True)
        all_edges = cv2.approxPolyDP(con, 0.018 * perimeter, True)
        if len(all_edges) == 4:
            x, y, a, b = cv2.boundingRect(con)
            #plate = image[y:y+b, x:x+a]
            #plate = cv2.rectangle(image, (x, y), (x + a, y + b), (0, 255, 0))
            all_plates.append(all_edges)
            #chosen = all_edges
            #break

    mask = np.zeros(image_grey.shape, np.uint8)
    if len(all_plates) == 0:
        return mask
    new_image = cv2.drawContours(mask, all_plates, 0, 255, -1)
    return mask



