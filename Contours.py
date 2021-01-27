import numpy as np
import cv2
import Checker


# FIND CONTOURS

# Input : Binary image of edges
# Output : List of objects, objects are defined by its corners

def find_contours(edges):
    contours = []
    current = np.argmax(edges)
    while edges[current // len(edges[0])][current % len(edges[0])] != 0:
        contour = continue_contour(edges, [current // len(edges[0]), current % len(edges[0])])
        if len(contour) > 60:
            contours.append(contour)
        current = np.argmax(edges)

    contours = [
        orient_helper(cv2.boxPoints(cv2.minAreaRect(np.array([[[point[1], point[0]]] for point in contours[i]])))) for i
        in range(len(contours))]

    return contours


def orient_helper(box):
    a, b, c, d = Checker.orient_corners((box[0], box[1], box[2], box[3]))
    return np.array([[a], [b], [c], [d]], dtype=np.int32)


def check_contour_list(point, contours):
    return True not in [point in contour for contour in contours]


def continue_contour(edges, point):
    found = []
    queue = [point]
    edges[point[0]][point[1]] = 0
    while len(queue) != 0:
        current = queue.pop(0)
        found.append(current)
        for dy, dx in {(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)}:
            new = [current[0] + dy, current[1] + dx]
            if 0 <= current[0] + dy < len(edges) \
                    and 0 <= current[1] + dx < len(edges[0]) \
                    and edges[current[0] + dy][current[1] + dx] == 255:
                edges[new[0]][new[1]] = 0
                queue.append(new)
    return found
