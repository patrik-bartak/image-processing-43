import time
from datetime import datetime

import cpuinfo
import cv2
import numpy as np

import Localization
import Recognize
import pattern_error_correction
import individual_format_verification

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
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    time_start = int(round(time.time()))

    cap = cv2.VideoCapture(file_path)
    # count = 0  # for file saving
    plate_strings = []  # program output

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imshow('input', frame)
        cv2.waitKey(1)
        plates = Localization.plate_detection(frame)
        # if localization does not find anything
        if plates is None or len(plates) == 0:
            continue

        for i in range(0, len(plates)):

            plate, string = Recognize.segment_and_recognize(plates[i])
            # if recognition fails to recognize a plate
            if string is None:
                print("Characters not recognized")
                continue
            # if a plate is localized and recognized, do some format verification
            string = individual_format_verification.verify_format(string, False)
            if string is None:
                print("Recognized plate invalid format")
                continue

            # https://stackoverflow.com/questions/43830131/combine-more-than-1-opencv-images-and-show-them-in-cv2-imshow-in-opencv-python
            # count += 1  # for file saving
            # cv2.imwrite("images/out/img-{}.jpg".format(count), plate)
            # for displaying binary images
            if not np.max(plate) == 255:
                plate = plate * 255

            plate_strings.append(string)
            print(string)

            cv2.imshow('plate', plate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # perform some batch error correction on the complete output
    final_output = pattern_error_correction.correct_errors(plate_strings)

    total_time_taken = '* TOTAL Time taken: {} s.'.format(int(round(time.time())) - time_start)
    print(total_time_taken)

    # toggle writing to file
    write_to_file = True
    if write_to_file:
        dttime = datetime.today().strftime('date-%d-%m-%Y_time-%H-%M-%S')
        f = open("out/recognized_plates/{}.txt".format(dttime), "x")
        f.write("Date executed: {}\nTime taken: {}\nRaw num. plates recognized: {}\nCPU used: {}\n\n"
                .format(dttime, total_time_taken, len(final_output), cpuinfo.get_cpu_info()['brand_raw']))
        f.write("\n".join(final_output))
        f.close()

    cap.release()
    cv2.destroyAllWindows()
