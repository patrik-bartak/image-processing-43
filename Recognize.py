import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

matches = [cv2.imread("SameSizeLetters/1.bmp"),
           cv2.imread("SameSizeLetters/2.bmp"),
           cv2.imread("SameSizeLetters/3.bmp"),
           cv2.imread("SameSizeLetters/4.bmp"),
           cv2.imread("SameSizeLetters/5.bmp"),
           cv2.imread("SameSizeLetters/6.bmp"),
           cv2.imread("SameSizeLetters/7.bmp"),
           cv2.imread("SameSizeLetters/8.bmp"),
           cv2.imread("SameSizeLetters/9.bmp"),
           cv2.imread("SameSizeLetters/10.bmp"),
           cv2.imread("SameSizeLetters/11.bmp"),
           cv2.imread("SameSizeLetters/12.bmp"),
           cv2.imread("SameSizeLetters/13.bmp"),
           cv2.imread("SameSizeLetters/14.bmp"),
           cv2.imread("SameSizeLetters/15.bmp"),
           cv2.imread("SameSizeLetters/16.bmp"),
           cv2.imread("SameSizeLetters/17.bmp"),
           cv2.imread("SameSizeNumbers/0.bmp"),
           cv2.imread("SameSizeNumbers/1.bmp"),
           cv2.imread("SameSizeNumbers/2.bmp"),
           cv2.imread("SameSizeNumbers/3.bmp"),
           cv2.imread("SameSizeNumbers/4.bmp"),
           cv2.imread("SameSizeNumbers/5.bmp"),
           cv2.imread("SameSizeNumbers/6.bmp"),
           cv2.imread("SameSizeNumbers/7.bmp"),
           cv2.imread("SameSizeNumbers/8.bmp"),
           cv2.imread("SameSizeNumbers/9.bmp")]

index = ["B", "D", "F", "G", "H", "J", "K",
         "L", "M", "N", "P", "R", "S", "T",
         "V", "X", "Z", "0", "1", "2", "3",
         "4", "5", "6", "7", "8", "9"]

for i in range(len(matches)):
    match = cv2.cvtColor(matches[i], cv2.COLOR_BGR2GRAY)
    projected_matches = np.sum(match, axis=0) / 255

    space_eplison = 8 + np.min(projected_matches)

    # replace "spaces" with grey pixels
    match[:, projected_matches < space_eplison] = np.array(100)

    index_list = [index for index, val in enumerate(projected_matches) if val >= space_eplison]

    matches[i] = matches[i][:, index_list[0]:index_list[-1]]

"""
In this file, you will define your own segment_and_recognize function.
To do:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
    3. Recognize the character by comparing the distances
Inputs:(One)
    1. plate_imgs: cropped plate images by Localization.plate_detection function
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
    1. recognized_plates: recognized plate characters
    type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
    You may need to define other functions.
"""


def segment_and_recognize(plate):
    proj = np.sum(plate, axis=0) / 255

    space_eplison = 8 + np.min(proj)
    epsilon = 45

    # replace "spaces" with grey pixels
    plate[:, proj < space_eplison] = np.array(100)

    index_list = [index for index, val in enumerate(proj) if val < space_eplison]

    # First attempt at segmenting the chars into a chars array
    chars = []
    for i in range(0, len(index_list) - 1):
        if index_list[i] + 1 == index_list[i + 1]:
            continue

        temp = cv2.resize(plate[:, index_list[i]: index_list[i + 1]], (100, 85))
        if np.mean(temp) < epsilon:
            continue
        chars.append(temp)

    print("Number of chars segmented: {}".format(len(chars)))

    # print bar plot of vertically projected plate
    # plt.bar(np.arange(500), proj)
    # plt.show()
    if (len(chars) == 0):
        return None, None

    return np.concatenate(chars, axis=1), recognize(chars)


def recognize(characters):
    epsilon = 0.1  # percentage of match
    result = ""
    for char in characters:
        max = 0
        chosen = None
        for i in range(len(matches)):
            match = cv2.resize(cv2.cvtColor(matches[i], cv2.COLOR_BGR2GRAY), (100, 85))
            char = cv2.threshold(cv2.resize(char, (100, 85)), 50, 255, cv2.THRESH_BINARY)[1]
            percent = np.sum(cv2.bitwise_and(char, match)) / np.sum(cv2.bitwise_or(char, match))
            #percent -= np.sum(cv2.bitwise_xor(char, match)) / np.sum(cv2.bitwise_or(char, match))
            if percent > max:
                max = percent
                chosen = index[i]
        if max > epsilon:
            result += chosen
        else:
            result += "_"
    return result
