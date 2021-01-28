import cv2
import numpy as np

from read_templates import get_templates

standard = [cv2.imread('imageprocessingcourse/templates/LettersStandard/1.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/2.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/3.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/4.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/5.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/6.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/7.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/8.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/9.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/10.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/11.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/12.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/13.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/14.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/15.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/16.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/LettersStandard/17.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/0.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/1.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/2.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/3.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/4.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/5.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/6.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/7.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/8.bmp', cv2.IMREAD_GRAYSCALE),
            cv2.imread('imageprocessingcourse/templates/NumbersStandard/9.bmp', cv2.IMREAD_GRAYSCALE)]

shear_left = [cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/1.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/2.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/3.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/4.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/5.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/6.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/7.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/8.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/9.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/10.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/11.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/12.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/13.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/14.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/15.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/16.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/LettersShearedLeft/17.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/0.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/1.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/2.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/3.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/4.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/5.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/6.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/7.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/8.bmp', cv2.IMREAD_GRAYSCALE),
              cv2.imread('imageprocessingcourse/templates/NumbersShearedLeft/9.bmp', cv2.IMREAD_GRAYSCALE)]

shear_right = [cv2.imread('imageprocessingcourse/templates/LettersShearedRight/1.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/2.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/3.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/4.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/5.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/6.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/7.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/8.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/9.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/10.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/11.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/12.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/13.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/14.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/15.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/16.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/LettersShearedRight/17.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/0.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/1.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/2.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/3.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/4.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/5.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/6.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/7.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/8.bmp', cv2.IMREAD_GRAYSCALE),
               cv2.imread('imageprocessingcourse/templates/NumbersShearedRight/9.bmp', cv2.IMREAD_GRAYSCALE)]


templates = [standard, shear_left, shear_right]
index = ["B", "D", "F", "G", "H", "J", "K",
         "L", "M", "N", "P", "R", "S", "T",
         "V", "X", "Z", "0", "1", "2", "3",
         "4", "5", "6", "7", "8", "9"]

for i in range(len(templates)):
    for j in range(len(templates[0])):
        templates[i][j] = cv2.threshold(cv2.resize(templates[i][j], (60, 85)), 120, 1, cv2.THRESH_BINARY)[1]

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


def segment_and_recognize(original_plate):
    plate = cv2.threshold(original_plate, 120, 1, cv2.THRESH_BINARY)[1]
    proj = np.sum(plate, axis=0)

    space_eplison = 4  # A space must have less than 4 pixels on its vertical projection
    epsilon = 100  # Only characters with more than epsilon pixels will be kept

    # replace "spaces" with grey pixels
    # plate[:, proj < space_eplison] = np.array(100)

    index_list = [index for index, val in enumerate(proj) if val < space_eplison]

    # First attempt at segmenting the chars into a chars array
    chars = []
    for i in range(0, len(index_list) - 1):
        if index_list[i] + 1 == index_list[i + 1]:
            continue

        temp = plate[:, index_list[i]: index_list[i + 1]]
        if np.sum(temp) < epsilon:  # sum instead of mean to better keep hyphens
            continue
        temp = cv2.resize(temp, (60, 85))
        # print(np.sum(temp))

        chars.append(temp)

    # print("Number of chars segmented: {}".format(len(chars)))

    # print bar plot of vertically projected plate
    # plt.bar(np.arange(500), proj)
    # plt.show()
    if len(chars) == 0:
        return None, None

    crop_horizontal(chars)

    # return np.concatenate(chars, axis=1), recognize(chars)
    plate_img, plate_string = np.concatenate(chars, axis=1), recognize(chars)

    return plate_img, plate_string


def recognize(characters):
    epsilon = 0.2  # percentage of match
    result = ""
    for char in characters:
        if np.sum(char) < 900:
            result += "-"
            continue
        max = 0
        chosen = None
        for i in range(len(templates)):
            for j in range(len(templates[0])):
                match = cv2.resize(templates[i][j], (60, 85))
                char = cv2.resize(char, (60, 85))
                percent = np.sum(cv2.bitwise_and(char, match)) / np.sum(cv2.bitwise_or(char, match))
                # percent -= np.sum(cv2.bitwise_xor(char, match)) / np.sum(cv2.bitwise_or(char, match))
                if percent > max:
                    max = percent
                    chosen = index[j]
        if max > epsilon:
            result += chosen
        else:
            result += "_"
    return result


def crop_horizontal(chars):
    for i in range(len(chars)):
        if np.sum(chars[i]) < 900:
            continue
        proj = np.sum(chars[i], axis=1)
        # chars[i][proj < 1, :] = np.array(100)

        index_list = [index for index, val in enumerate(proj) if val < 1]
        if len(index_list) == 0: continue
        segments = []
        sizes = []
        temp = chars[i][0: index_list[0]]
        if len(temp) > 0:
            segments.append(temp)
            sizes.append(len(temp) * len(temp[0]))
        temp = chars[i][index_list[len(index_list) - 1]: len(chars[i])]
        if len(temp) > 0:
            segments.append(temp)
            sizes.append(len(temp) * len(temp[0]))
        for j in range(0, len(index_list) - 1):
            if index_list[j] + 1 == index_list[j + 1]:
                continue
            temp = chars[i][index_list[j]: index_list[j + 1]]
            segments.append(temp)
            sizes.append(len(temp) * len(temp[0]))
        if len(segments) == 0: continue
        chars[i] = cv2.resize(segments[np.argmax(sizes)], (60, 85))
