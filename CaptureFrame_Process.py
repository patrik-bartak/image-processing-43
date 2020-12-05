import cv2
import os
import pandas as pd
import Localization
import Recognize
import numpy as np

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""
def CaptureFrame_Process(file_path, sample_frequency, save_path):
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = Localization.plate_detection(frame)

        if plates is None or len(plates) == 0:
            continue

        #https://stackoverflow.com/questions/43830131/combine-more-than-1-opencv-images-and-show-them-in-cv2-imshow-in-opencv-python
        imstack = None
        for plate in plates:
                    #im = cv2.resize(plate, (100, 500))
                    if imstack is None:
                        imstack = np.hstack(plate)
                    else:
                        imstack = np.hstack(imstack, plate)
        #im = plates[0]#cv2.resize(plates, (1000, 800))

        cv2.imshow('Resulting video', imstack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
