import time

import cv2
import numpy as np
import Canny
import Contours
import Checker

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

start = 0
show = True


def plate_detection(image, im_show):
    # Replace the below lines with your code.
    # https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python
    # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
    # https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv

    # https://tajanthan.github.io/cv/docs/anpr.pdf
    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    global start
    global show
    show = im_show

    start = int(round(time.time() * 1000))
    detected_plates = []
    original = image.copy()
    mask = get_yellow_mask(original)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    new_image = cv2.bitwise_and(original, original, mask=mask)
    corner_coords_arr = get_plates_by_bounding(new_image)
    if len(corner_coords_arr) == 0:
        return None

    for coords in corner_coords_arr:
        coords = np.reshape(coords, (4, 2))
        coords = Checker.orient_corners(coords)
        s_plate_w, s_plate_h = 500, 100
        p_mat = custom_get_transform_matrix(
            np.float32(coords),
            np.float32([[0, 0], [s_plate_w, 0], [s_plate_w, s_plate_h], [0, s_plate_h]])
        )
        plate = custom_reverse_warp_perspective(image, p_mat, (480, 640), (s_plate_h, s_plate_w))
        # plate = cv2.warpPerspective(image, p_mat, (s_plate_w, s_plate_h))
        print("Time needed: ", int(round(time.time() * 1000)) - start, ' ms.')
        g_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        plate = cv2.filter2D(np.float64(plate.copy()), -1, g_kernel).astype(np.uint8)
        # plate = cv2.resize(plate[5:95, 20:480], (500, 100))
        if show:
            cv2.imshow('1 - Projected', plate)
            cv2.waitKey(1)
        detected_plates.append(plate)
    return detected_plates


# accepts a black & white 255 image and performs edge cleanup using flood fill
def edge_cleanup_procedure(plate):
    if show:
        cv2.imshow('2 - binarized', plate)
        cv2.waitKey(1)
    kernel = np.ones((3, 3), np.uint8)
    # Erode, flood fill at edges, then dilate
    plate = cv2.erode(plate, kernel, iterations=1)
    if show:
        cv2.imshow('3 - eroded', plate)
        cv2.waitKey(1)
    # x = np.arange(100, dtype='2int32')
    # print(x)
    # flood_fill_points = [np.concatenate((np.zeros(500), np.ones(500)*99, np.arange(100), np.arange(100))),
    # np.concatenate((np.arange(500), np.arange(500), np.zeros(100), np.ones(100)*499))]
    h, w = np.shape(plate)
    for y in [0, h - 1]:
        for x in range(w):
            if plate[y, x] != 0:
                custom_flood_fill(plate, (y, x), 0)
    for x in [0, w - 1]:
        for y in range(h):
            if plate[y, x] != 0:
                custom_flood_fill(plate, (y, x), 0)
    if show:
        cv2.imshow('4 - floodfilled', plate)
        cv2.waitKey(1)
    plate = cv2.dilate(plate, kernel, iterations=1)
    if show:
        cv2.imshow('5 - dilated', plate)
        cv2.waitKey(1)
    return plate


def custom_flood_fill(plate, init, value):
    # https://www.geeksforgeeks.org/flood-fill-algorithm/
    custom_flood_fill_helper(plate, init, value, [])


def custom_flood_fill_helper(plate, init, value, q):
    h, w = np.shape(plate)
    old = plate[init]
    if old == value:
        return
    plate[init] = value
    q.append(init)
    while q:
        y, x = q.pop()
        if 0 <= y + 1 < h and 0 <= x < w and plate[y + 1, x] == old:
            plate[y + 1, x] = value
            q.append([y + 1, x])
        if 0 <= y - 1 < h and 0 <= x < w and plate[y - 1, x] == old:
            plate[y - 1, x] = value
            q.append([y - 1, x])
        if 0 <= y < h and 0 <= x + 1 < w and plate[y, x + 1] == old:
            plate[y, x + 1] = value
            q.append([y, x + 1])
        if 0 <= y < h and 0 <= x - 1 < w and plate[y, x - 1] == old:
            plate[y, x - 1] = value
            q.append([y, x - 1])


def binarize_255(plate, threshold):
    # Histogram equalization and threshold
    plate = cv2.cvtColor(plate.copy(), cv2.COLOR_BGR2GRAY)
    plate = custom_equalize_histogram(plate)
    if show:
        cv2.imshow('2.5 - equalized', plate)
        cv2.waitKey(1)
    lower = np.array(0, dtype="uint8")
    upper = np.array(threshold, dtype="uint8")
    mask = cv2.inRange(plate, lower, upper)
    return mask


def custom_equalize_histogram(plate):
    histogram, bins = np.histogram(plate, bins=np.arange(257))
    cumulative = np.cumsum(histogram)
    cum_norm = (((cumulative - np.min(cumulative)) / (np.max(cumulative) - np.min(cumulative))) * 255).astype("uint8")
    eq = cum_norm[np.reshape(plate, -1)]
    return np.reshape(eq, np.shape(plate))


def custom_get_transform_matrix(f, t):
    # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    b = np.zeros(9)
    b[8] = 1
    # x0, x1, x2, x3, y0, y1, y2, y3, x0p, x1p, x2p, x3p, y0p, y1p, y2p, y3p =
    a = [
        [-f[0, 0], -f[0, 1], -1, 0, 0, 0, f[0, 0] * t[0, 0], f[0, 1] * t[0, 0], t[0, 0]],
        [0, 0, 0, -f[0, 0], -f[0, 1], -1, f[0, 0] * t[0, 1], f[0, 1] * t[0, 1], t[0, 1]],
        [-f[1, 0], -f[1, 1], -1, 0, 0, 0, f[1, 0] * t[1, 0], f[1, 1] * t[1, 0], t[1, 0]],
        [0, 0, 0, -f[1, 0], -f[1, 1], -1, f[1, 0] * t[1, 1], f[1, 1] * t[1, 1], t[1, 1]],
        [-f[2, 0], -f[2, 1], -1, 0, 0, 0, f[2, 0] * t[2, 0], f[2, 1] * t[2, 0], t[2, 0]],
        [0, 0, 0, -f[2, 0], -f[2, 1], -1, f[2, 0] * t[2, 1], f[2, 1] * t[2, 1], t[2, 1]],
        [-f[3, 0], -f[3, 1], -1, 0, 0, 0, f[3, 0] * t[3, 0], f[3, 1] * t[3, 0], t[3, 0]],
        [0, 0, 0, -f[3, 0], -f[3, 1], -1, f[3, 0] * t[3, 1], f[3, 1] * t[3, 1], t[3, 1]],
        b.copy(),
    ]
    x = np.linalg.solve(a, b)
    res = np.zeros((3, 3))
    res[0] = x[0:3]
    res[1] = x[3:6]
    res[2] = x[6:9]
    return res


def custom_reverse_warp_perspective(image, p_mat, from_shape, to_shape):
    inv_mat = np.linalg.inv(p_mat)
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    new_img = np.zeros((to_h, to_w, 3))

    x = np.arange(0, to_w)
    y = np.arange(0, to_h)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)

    idx_new = np.matmul(inv_mat, [xx, yy, np.ones(to_w * to_h)])
    idx_new = (idx_new / idx_new[2]).astype(int)
    xx = idx_new[0]
    xx = np.where(xx < 0, 0, xx)
    xx = np.where(xx >= from_w, from_w - 1, xx)
    yy = idx_new[1]
    yy = np.where(yy < 0, 0, yy)
    yy = np.where(yy >= from_h, from_h - 1, yy)

    old = np.reshape([yy, xx], (2, to_h, to_w))

    new_img[:, :] = image[old[0, :, :], old[1, :, :]]

    return new_img


def get_yellow_mask(image):
    image_yellow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([8, 93, 100], dtype="uint8")
    upper_yellow = np.array([45, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(image_yellow, lower_yellow, upper_yellow)
    return mask_yellow


def get_plates_by_bounding(image):
    # https://www.youtube.com/watch?v=UgGLo_QRHJ8
    all_plates = []

    image_edges = Canny.canny(image, 0, 0)
    if show:
        cv2.imshow('canny', image_edges)
        cv2.waitKey(1)
    # return image_edges
    # print(np.mean(abs(image_edges - check)))
    # contours, _ = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_reduced = Contours.find_contours(image_edges)

    # print('Contours: ', len(contours))
    contours_reduced = sorted(contours_reduced, key=cv2.contourArea, reverse=True)[:5]
    temp = cv2.drawContours(image, contours_reduced, -1, (0, 255, 0), 1)
    if show:
        cv2.imshow('contours', temp)
        cv2.waitKey(1)

    for con in contours_reduced:
        # all_plates.append(all_edges)
        if Checker.check(con):
            all_plates.append(con)
    # (i, 4, 2) array of corner coordinates for i plates in the image
    return all_plates
