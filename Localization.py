import cv2
import numpy as np

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

    # https://tajanthan.github.io/cv/docs/anpr.pdf

    original = image.copy()
    mask_yellow = get_yellow_mask(image)
    mask = mask_yellow

    mask_loc = get_plates_by_bounding(image)
    # mask2 = get_plates_by_bounding(image, True)

    # mask = cv2.bitwise_or(mask, cv2.bitwise_and(mask2, mask_loc))
    mask = mask_loc

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #
    # result = cv2.bitwise_and(original, original, mask=mask)

    result = [cv2.bitwise_or(original, original, mask=mask)]

    return result


def get_yellow_mask(image):
    image_yellow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([8, 93, 100], dtype="uint8")
    upper_yellow = np.array([45, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(image_yellow, lower_yellow, upper_yellow)
    return mask_yellow


def get_plates_by_bounding(image):
    # https://www.youtube.com/watch?v=UgGLo_QRHJ8
    all_plates = []
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    image_blur = cv2.bilateralFilter(image_grey, 4, 150, 150)
    image_edges = cv2.Canny(image_blur, 30, 150)
    contours, _ = cv2.findContours(image_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_reduced = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for con in contours_reduced:
        perimeter = cv2.arcLength(con, True)
        all_edges = cv2.approxPolyDP(con, 0.018 * perimeter, True)
        if len(all_edges) == 4 and check_distances(all_edges):
            all_plates.append(all_edges)

    mask = np.zeros(image_grey.shape, np.uint8)
    if len(all_plates) == 0:
        return mask
    new_image = cv2.drawContours(mask, all_plates, 0, 255, -1)
    return mask


def check_distances(points, epsilon=45):
    edges = []
    for i in range(4):
        A = points[i][0]
        next = i + 1
        if i == 3:
            next = 0
        B = points[next][0]
        edges.append([B[0] - A[0], B[1] - A[1]])

    dist = []
    for i in range(4):
        dist.append(np.linalg.norm(edges[i]))

    return check_ratio(dist) and angle_check(edges) and check_area(dist)


def check_area(dist, epsilon = 500):
    return dist[0] * dist[1] >= epsilon


def angle_check(vectors, epsilon=0.5):
    return get_angle_dot(vectors[0], vectors[1]) < epsilon and get_angle_dot(vectors[2], vectors[3]) < epsilon


def get_angle_dot(vec1, vec2):
    unit1 = vec1 / np.linalg.norm(vec1)
    unit2 = vec2 / np.linalg.norm(vec2)
    return abs(np.dot(unit1, unit2))


def check_ratio(dist, epsilon=3):
    ratio = 5
    return abs(dist[0] / dist[1] - ratio) < epsilon and abs(dist[2] / dist[3] - ratio) < epsilon
