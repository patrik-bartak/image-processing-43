import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

	# replace "spaces" with grey pixels
	plate[:, proj < space_eplison] = np.array(100)

	index_list = [index for index, val in enumerate(proj) if val < space_eplison]

	# First attempt at segmenting the chars into a chars array
	chars = []
	for i in range(0, len(index_list) - 1):
		if index_list[i] + 1 == index_list[i + 1]:
			continue
		chars.append(plate[:, index_list[i]: index_list[i + 1]])

	print("Number of chars segmented: {}".format(len(chars)))

	# print bar plot of vertically projected plate
	# plt.bar(np.arange(500), proj)
	# plt.show()

	return plate