import cv2
import numpy as np


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
