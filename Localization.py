import time

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

start = 0
imshow_on = True


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
    mask = get_yellow_mask(original)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    new_image = cv2.bitwise_and(original, original, mask=mask)
    corner_coords_arr = get_plates_by_bounding(new_image)
    if len(corner_coords_arr) == 0:
        return

    for coords in corner_coords_arr:
        coords = np.reshape(coords, (4, 2))
        coords = orient_corners(coords)
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
        if imshow_on:
            cv2.imshow('1 - Projected', plate)
            cv2.waitKey(1)
        detected_plates.append(plate)
    return detected_plates


# accepts a black & white 255 image and performs edge cleanup using flood fill
def edge_cleanup_procedure(plate):
    if imshow_on:
        cv2.imshow('2 - binarized', plate)
        cv2.waitKey(1)
    kernel = np.ones((3, 3), np.uint8)
    # Erode, flood fill at edges, then dilate
    plate = cv2.erode(plate, kernel, iterations=1)
    if imshow_on:
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
    if imshow_on:
        cv2.imshow('4 - floodfilled', plate)
        cv2.waitKey(1)
    plate = cv2.dilate(plate, kernel, iterations=1)
    if imshow_on:
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
    if imshow_on:
        cv2.imshow('2.5 - equalized', plate)
        cv2.waitKey(1)
    lower = np.array(0, dtype="uint8")
    upper = np.array(threshold, dtype="uint8")
    mask = cv2.inRange(plate, lower, upper)
    return mask


def custom_equalize_histogram(plate):
    histogram, bins = np.histogram(plate, bins=np.arange(256))
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


def orient_corners(c):
    tl, tr, br, bl = c
    # Flip plate if mirrored along y axis
    if tl[0] > tr[0]:
        tl, tr = tr, tl
    if bl[0] > br[0]:
        bl, br = br, bl
    if tl[0] > br[0]:
        tl, br = br, tl
    if bl[0] > tr[0]:
        bl, tr = tr, bl
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
    if imshow_on:
        cv2.imshow('canny', image_edges)
        cv2.waitKey(1)
    # return image_edges
    # print(np.mean(abs(image_edges - check)))
    # contours, _ = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_reduced = find_contours(image_edges)

    # print('Contours: ', len(contours))
    contours_reduced = sorted(contours_reduced, key=cv2.contourArea, reverse=True)[:5]
    temp = cv2.drawContours(image, contours_reduced, -1, (0, 255, 0), 1)
    cv2.imshow('contours', temp)
    cv2.waitKey(1)

    for con in contours_reduced:
        perimeter = cv2.arcLength(con, True)
        all_edges = cv2.approxPolyDP(con, 0.018 * perimeter, True)
        # all_plates.append(all_edges)
        if len(all_edges) == 4 and check(all_edges):
            all_plates.append(all_edges)
    # (i, 4, 2) array of corner coordinates for i plates in the image
    cv2.imshow('chosen', cv2.drawContours(image, all_plates, -1, (0, 255, 0), 1))
    cv2.waitKey(1)
    return all_plates


def print_diff(arr1, arr2):
    for i in range(len(arr1)):
        res = ''
        for j in range(len(arr1[0])):
            res += str(np.int64(arr1[i][j]) - np.int64(arr2[i][j]))
            res += ', '
        print(res)


# FIND CONTOURS

# Input : Binary image of edges
# Output : List of objects, objects are defined by its corners

def find_contours(edges):
    contours = []
    last = None
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            current = [i, j]
            if last is not None and edges[last[0]][last[1]] == 0 and edges[current[0]][current[1]] == 255 \
                    and check_contour_list(current, contours):
                found = [current in contour for contour in contours]

                if True not in found:
                    temp = continue_contour(edges, current)
                    if len(temp) >= 60:
                        contours.append(temp)
            last = current

    contours = [orient_helper(cv2.boxPoints(cv2.minAreaRect(np.array([[[point[1], point[0]]] for point in contours[i]])))) for i in range(len(contours))]

    return contours


def orient_helper(box):
    a, b, c, d = orient_corners((box[0], box[1], box[2], box[3]))
    return np.array([[a], [b], [c], [d]], dtype=np.int32)


def check_contour_list(point, contours):
    return True not in [point in contour for contour in contours]


def continue_contour(edges, point):
    found = []
    queue = [point]
    while len(queue) != 0:
        current = queue.pop(0)
        found.append(current)
        for dy, dx in {(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)}:
            new = [current[0] + dy, current[1] + dx]
            if 0 <= current[0] + dy < len(edges) \
                    and 0 <= current[1] + dx < len(edges[0]) \
                    and edges[current[0] + dy][current[1] + dx] == 255 \
                    and check_contour_list(new, [found]) and check_contour_list(new, [queue]):
                queue.insert(0, new)
    return found


# CANNY

# https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
def canny(image, lower, upper):
    # Making the image greyscale
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    # Noise reduction using Gaussian kernel - step 1 of Canny
    image_f = cv2.bilateralFilter(image_grey, 5, 150, 150)  # TODO
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
    return cv2.filter2D(image, -1, kernel)


# OTHER HELPER FUNCTIONS

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


def check_ratio(dist, epsilon=1.5):
    ratio = 4
    return abs(max(dist[1], dist[0]) / min(dist[1], dist[0]) - ratio) < epsilon and abs(
        max(dist[3], dist[2]) / min(dist[3], dist[2]) - ratio) < epsilon
