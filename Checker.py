import numpy as np


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


def print_diff(arr1, arr2):
    for i in range(len(arr1)):
        res = ''
        for j in range(len(arr1[0])):
            res += str(np.int64(arr1[i][j]) - np.int64(arr2[i][j]))
            res += ', '
        print(res)
