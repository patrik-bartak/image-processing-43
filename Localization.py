import cv2
import numpy as np
from scipy import signal
import time

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
flood_fill_points = [(0, 0), (479, 0), (479, 639), (0, 639), (239, 0), (479, 319), (239, 639), (0, 319)]

def plate_detection(image):
    # Replace the below lines with your code.
    # https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python
    # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
    # https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv

    # https://tajanthan.github.io/cv/docs/anpr.pdf
    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    global start
    start = int(round(time.time() * 1000))

    detected_plates = []

    original = image.copy()
    epsilon = 0.1
    corner_coords_arr = get_plates_by_bounding(image)

    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(mask, corner_coords_arr, 0, 255, -1)

    if mask.mean() < epsilon or len(corner_coords_arr) == 0:
        mask = get_yellow_mask(original)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        new_image = cv2.bitwise_and(original, original, mask=mask)
        corner_coords_arr = get_plates_by_bounding(new_image)
        if len(corner_coords_arr) == 0:
            return

    for coords in corner_coords_arr:
        coords = np.reshape(coords, (4, 2))
        coords = orient_corners(coords)
        M = cv2.getPerspectiveTransform(np.float32(coords), np.float32([[0, 0], [480, 0], [480, 640], [0, 640]]))
        plate = cv2.warpPerspective(image, M, (480, 640))
        print("Time needed: ", int(round(time.time() * 1000)) - start, ' ms.')
        plate = binarize(plate)
        kernel = np.ones((3, 3), np.uint8)
        # Erode, flood fill at edges, then dilate
        plate_eroded = cv2.erode(plate, kernel, iterations=2)
        for a, b in flood_fill_points:
            if plate_eroded[b, a] != 0:
                cv2.floodFill(plate_eroded, None, (a, b), 0, 200, 200)
        plate = cv2.dilate(plate_eroded, kernel, iterations=2)
        print(np.count_nonzero(mask) / np.array(mask).size)
        detected_plates.append(plate)
    return detected_plates


def binarize(plate):
    # Contrast stretch and threshold
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate = cv2.equalizeHist(plate)
    lower = np.array(0, dtype="uint8")
    upper = np.array(60, dtype="uint8")
    mask = cv2.inRange(plate, lower, upper)
    return mask


def orient_corners(c):
    tl, tr, br, bl = c
    # Flip plate if mirrored along y axis
    if tl[0] > tr[0]:
        tl, tr = tr, tl
    if bl[0] > br[0]:
        bl, br = br, bl
    # Flip plate if mirrored along x axis
    if tl[1] > bl[1]:
        tl, bl = bl, tl
    if tr[1] > br[1]:
        tr, br = br, tr
    return tl, tr, br, bl


def crop(image, mask):
    return get_segment_crop(image, mask=mask)


def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def get_yellow_mask(image):
    image_yellow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([8, 93, 100], dtype="uint8")
    upper_yellow = np.array([45, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(image_yellow, lower_yellow, upper_yellow)
    return mask_yellow


def get_plates_by_bounding(image):
    # https://www.youtube.com/watch?v=UgGLo_QRHJ8
    all_plates = []

    image_edges = canny(image, 0, 0)
    #return image_edges
    # print(np.mean(abs(image_edges - check)))
    contours, _ = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('Contours: ', len(contours))
    contours_reduced = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for con in contours_reduced:
        perimeter = cv2.arcLength(con, True)
        all_edges = cv2.approxPolyDP(con, 0.018 * perimeter, True)
        # all_plates.append(all_edges)
        if len(all_edges) == 4 and check(all_edges):
            all_plates.append(all_edges)
    # (i, 4, 2) array of corner coordinates for i plates in the image
    return all_plates


def print_diff(arr1, arr2):
    for i in range(len(arr1)):
        res = ''
        for j in range(len(arr1[0])):
            res += str(np.int64(arr1[i][j]) - np.int64(arr2[i][j]))
            res += ', '
        print(res)


# https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
def canny(image, lower, upper):
    # Making the image greyscale
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    # Noise reduction using Gaussian kernel - step 1 of Canny
    image_f = cv2.bilateralFilter(image_grey, 5, 150, 150)
    # check = cv2.Canny(image_f, lower, upper)
    # Gradient calculation - step 2 of Canny
    gradient, direction = get_gradient(image_f)
    # print("Gradient: ", int(round(time.time() * 1000)) - start, ' ms.')
    # Non-maximum suppression - step 3 of Canny
    gradient_thin = non_max_suppression(gradient, direction)
    # print("Thinning: ", int(round(time.time() * 1000)) - start, ' ms.')
    # Double threshold - step 4 of Canny
    edges = apply_thresholds(gradient_thin, lower, upper)
    # print("Edge thresholds: ", int(round(time.time() * 1000)) - start, ' ms.')
    result = edge_running(edges)
    # print("Edge running: ", int(round(time.time() * 1000)) - start, ' ms.')
    return np.uint8(result)


def non_max_suppression(gradient, d, lower=10):
    height, width = gradient.shape
    res = np.zeros((height, width))
    # d[np.where(-np.pi/8 <= d <= np.pi/8)]
    # rows, cols = np.where(np.abs(d) > np.pi * 3 / 8)

    # rows, cols = np.where(np.abs(d) <= np.pi / 8)
    # res[rows, cols] = gradient[rows, cols]

    # for i in range(1, h - 1):
    #     for j in range(1, w - 1):
    #         if gradient[i, j] >= gradient[i, j+1] and gradient[i, j] >= gradient[i, j-1]:
    #             res[i, j] = gradient[i, j]
    #         else:
    #             res[i, j] = 0

    # print(res)
    # row, col = np.where(gradient >= 0)
    #
    # for index in range(len(row)):
    #     i = row[index]
    #     j = col[index]
    #     if i == 0 or i == height - 1 or j == 0 or j == width - 1:
    #         continue
    #     if np.abs(d[i, j]) <= np.pi / 8:
    #         res[i, j] = gradient[i, j] if gradient[i, j] >= gradient[i, j + 1] \
    #                                       and gradient[i, j] >= gradient[i, j - 1] else 0
    #     elif np.abs(d[i, j]) >= np.pi * 3 / 8:
    #         res[i, j] = gradient[i, j] if gradient[i, j] >= gradient[i + 1, j] \
    #                                       and gradient[i, j] >= gradient[i - 1, j] else 0
    #     elif np.pi * 1 / 8 <= d[i, j] <= np.pi * 3 / 8:
    #         res[i, j] = gradient[i, j] if gradient[i, j] >= gradient[i + 1, j - 1] \
    #                                       and gradient[i, j] >= gradient[i - 1, j + 1] else 0
    #     elif np.pi * -1 / 8 >= d[i, j] >= np.pi * -3 / 8:
    #         res[i, j] = gradient[i, j] if gradient[i, j] >= gradient[i + 1, j + 1] \
    #                                       and gradient[i, j] >= gradient[i - 1, j - 1] else 0
    # return res

    d_tilt_down = d.copy()
    d_tilt_down[np.logical_and(np.pi * -1 / 8 >= d_tilt_down, d_tilt_down >= np.pi * -3 / 8)] = 255
    d_tilt_down[d_tilt_down != 255] = 0
    d_tilt_down = np.uint8(d_tilt_down)

    d_tilt_up = d.copy()
    d_tilt_up[np.logical_and(np.pi * 1 / 8 <= d_tilt_up, d_tilt_up <= np.pi * 3 / 8)] = 255
    d_tilt_up[d_tilt_up != 255] = 0
    d_tilt_up = np.uint8(d_tilt_up)

    d_up = d.copy()
    d_up[np.abs(d_up) >= np.pi * 3 / 8] = 255
    d_up[d_up != 255] = 0
    d_up = np.uint8(d_up)

    d_hor = d.copy()
    d_hor[np.abs(d_hor) <= np.pi / 8] = 255
    d_hor[d_hor != 255] = 0
    d_hor = np.uint8(d_hor)

    epsilon = 0.2

    # -1 0 0
    # 0 1 0
    # 0 0 0
    tiltDown_a = np.zeros((3, 3))
    tiltDown_a[0][0] = -1
    tiltDown_a[1][1] = 1 + epsilon

    # 0 0 0
    # 0 1 0
    # 0 0 -1
    tiltDown_b = np.zeros((3, 3))
    tiltDown_b[2][2] = -1
    tiltDown_b[1][1] = 1 + epsilon

    # 0 0 0
    # 0 1 0
    # -1 0 0
    tiltUp_a = np.zeros((3, 3))
    tiltUp_a[2][0] = -1
    tiltUp_a[1][1] = 1 + epsilon

    # 0 0 -1
    # 0 1 0
    # 0 0 0
    tiltUp_b = np.zeros((3, 3))
    tiltUp_b[0][2] = -1
    tiltUp_b[1][1] = 1 + epsilon

    # 0 0 0
    # 0 1 0
    # 0 -1 0
    up_a = np.zeros((3, 3))
    up_a[2][1] = -1
    up_a[1][1] = 1 + epsilon

    # 0 -1 0
    # 0 1 0
    # 0 0 0
    up_b = np.zeros((3, 3))
    up_b[0][1] = -1
    up_b[1][1] = 1 + epsilon

    # 0 0 0
    # 0 1 -1
    # 0 0 0
    hor_a = np.zeros((3, 3))
    hor_a[1][2] = -1
    hor_a[1][1] = 1 + epsilon

    # 0 0 0
    # -1 1 0
    # 0 0 0
    hor_b = np.zeros((3, 3))
    hor_b[1][0] = -1
    hor_b[1][1] = 1 + epsilon

    tD_a = cv2.filter2D(np.float64(gradient.copy()), -1, tiltDown_a)
    tU_a = cv2.filter2D(np.float64(gradient.copy()), -1, tiltUp_a)
    u_a = cv2.filter2D(np.float64(gradient.copy()), -1, up_a)
    h_a = cv2.filter2D(np.float64(gradient.copy()), -1, hor_a)

    tD_b = cv2.filter2D(np.float64(gradient.copy()), -1, tiltDown_b)
    tU_b = cv2.filter2D(np.float64(gradient.copy()), -1, tiltUp_b)
    u_b = cv2.filter2D(np.float64(gradient.copy()), -1, up_b)
    h_b = cv2.filter2D(np.float64(gradient.copy()), -1, hor_b)

    tD_a[tD_a > 0] = 255
    tD_a[tD_a <= 0] = 0

    tD_b[tD_b > 0] = 255
    tD_b[tD_b <= 0] = 0

    tU_a[tU_a > 0] = 255
    tU_a[tU_a <= 0] = 0

    tU_b[tU_b > 0] = 255
    tU_b[tU_b <= 0] = 0

    u_a[u_a > 0] = 255
    u_a[u_a <= 0] = 0

    u_b[u_b > 0] = 255
    u_b[u_b <= 0] = 0

    h_a[h_a > 0] = 255
    h_a[h_a <= 0] = 0

    h_b[h_b > 0] = 255
    h_b[h_b <= 0] = 0

    tD = cv2.bitwise_and(tD_a, tD_b)
    tU = cv2.bitwise_and(tU_a, tU_b)
    u = cv2.bitwise_and(u_a, u_b)
    h = cv2.bitwise_and(h_a, h_b)

    final1 = cv2.bitwise_and(tD, tD, mask=d_tilt_down)
    final2 = cv2.bitwise_and(tU, tU, mask=d_tilt_up)
    final3 = cv2.bitwise_and(u, u, mask=d_up)
    final4 = cv2.bitwise_and(h, h, mask=d_hor)

    final = np.uint8(cv2.bitwise_or(cv2.bitwise_or(final1, final2), cv2.bitwise_or(final3, final4)))
    result = np.uint8(cv2.bitwise_and(gradient.copy(), gradient.copy(), mask=final))
    result[0, :] = 0
    result[:, 0] = 0
    result[height - 1, :] = 0
    result[:, width - 1] = 0
    return result


def apply_thresholds(image, lower=5, upper=20):
    _, mask_weak = cv2.threshold(image.copy(), lower, upper - 1, cv2.THRESH_TOZERO)
    _, mask_strong = cv2.threshold(image.copy(), upper, 255, cv2.THRESH_BINARY)
    return mask_weak + mask_strong


def edge_running(edges):
    kernel = np.ones((3, 3))
    first = cv2.filter2D(edges, -1, kernel)
    second = np.clip(first, a_min=0, a_max=255)
    res = np.uint8(second)
    final = np.uint8(cv2.threshold(res, 200, 255, cv2.THRESH_BINARY)[1])
    temp = np.uint8(cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)[1])
    return cv2.bitwise_and(final, temp)


def get_gradient(image):
    # Sobel gradient in x and y direction
    g_x, g_y = apply_sobel(image)
    # Gradient magnitude
    g = np.uint8(np.sqrt(np.power(g_x, 2) + np.power(g_y, 2)))
    # Gradient orientation
    g_x[g_x == 0] = 0.0001
    theta = np.arctan(g_y / g_x)
    return g, theta


def apply_sobel(image):
    kernel_y = np.zeros((3, 3))
    kernel_y[0][0], kernel_y[0][2] = -1, -1
    kernel_y[2][0], kernel_y[2][2] = 1, 1
    kernel_y[0][1], kernel_y[2][1] = -2, 2

    kernel_x = np.zeros((3, 3))
    kernel_x[0][0], kernel_x[0][2] = 1, -1
    kernel_x[2][0], kernel_x[2][2] = 1, -1
    kernel_x[1][0], kernel_x[1][2] = 2, -2
    image2 = np.float64(image)
    image3 = np.float64(image)
    return conv2d(image2, kernel_x), conv2d(image3, kernel_y)


# https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
def conv2d(image, kernel):
    return cv2.filter2D(image, -1, kernel)#signal.convolve2d(image, kernel, mode='same')


def check(points):
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


def check_area(dist, epsilon=500):
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
