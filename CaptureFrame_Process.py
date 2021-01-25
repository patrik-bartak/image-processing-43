import time
import difflib
from datetime import datetime

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


def CaptureFrame_Process(file_path, sample_frequency, save_path, show):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    time_start = int(round(time.time()))

    cap = cv2.VideoCapture(file_path)
    # count = 0  # for file saving
    plate_strings = []  # program output

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if show:
            cv2.imshow('input', frame)
            cv2.waitKey(1)
        t = 0
        while t < 10:
            cap.read()
            t += 1

        colour_plates = Localization.plate_detection(frame, show)
        # if localization does not find anything
        if colour_plates is None or len(colour_plates) == 0:
            continue

        for i in range(0, len(colour_plates)):
            thresholds = [40, 60, 70, 80, 100]
            while True:
                for j in range(len(thresholds)):
                    binarized = Localization.binarize_255(colour_plates[i], thresholds[j])
                    clean_plate = Localization.edge_cleanup_procedure(binarized)
                    # recognize the plate characters
                    plate, string = Recognize.segment_and_recognize(clean_plate)
                    # if a plate is localized and recognized, do some format verification
                    string = individual_format_verification.verify_format(string, True)
                    if string is not None:
                        break
                if string is not None or string is None and np.shape(colour_plates[i])[0] < 80:
                    break
                elif string is None:
                    colour_plates[i] = colour_plates[i][10:-10, 20:-20]
                    print("Invalid format, cropping image...")
                    continue

            if string is None:
                print("Recognized plate invalid format")
                continue
            # https://stackoverflow.com/questions/43830131/combine-more-than-1-opencv-images-and-show-them-in-cv2-imshow-in-opencv-python
            # count += 1  # for file saving
            # cv2.imwrite("images/out/img-{}.jpg".format(count), plate)
            # for displaying binary images
            if np.max(plate) == 1:
                plate = plate * 255

            plate_strings.append(string)
            print(string)

            if show:
                cv2.imshow('plate', plate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    total_time_taken = '* TOTAL Time taken: {} s.'.format(int(round(time.time())) - time_start)
    print(total_time_taken)

    postprocessing_time_start = int(round(time.time()))
    # perform some batch error correction on the complete output
    final_output = pattern_error_correction.correct_errors(plate_strings)

    postprocessing_time_taken = '* POSTPROCESSING Time taken: {} s.'.format(
        int(round(time.time())) - postprocessing_time_start)
    print(postprocessing_time_taken)

    # toggle writing to file
    write_to_file = True
    if write_to_file:
        arr_expected = open("plates.txt", "r").read().strip().splitlines()
        num_correct = np.sum([1 if expected in final_output else 0 for expected in arr_expected])
        num_incorrect = np.sum([1 if found not in arr_expected else 0 for found in final_output])

        path = "out/recognized_plates/{}.txt"
        file_out(final_output, path, num_correct, len(arr_expected), num_incorrect, total_time_taken)

        diff = difflib.unified_diff(arr_expected, final_output, fromfile='expected', tofile='actual', lineterm='', n=0)
        path = "out/plate_diffs/{}.txt"
        file_out(diff, path, num_correct, len(arr_expected), num_incorrect, total_time_taken)

    cap.release()
    cv2.destroyAllWindows()


def file_out(arr_output, path, num_correct, expected_num_plates, num_incorrect, total_time_taken):
    str_output = "\n".join(arr_output)
    dt = datetime.today().strftime('date-%d-%m-%Y_time-%H-%M')
    f = open(path.format(dt), "x")
    f.write("Date executed: {}\n"
            "Time taken: {}\n"
            "Num plates FOUND: {} out of {} expected\n"
            "Num plates INCORRECT: {} out of 0 expected\n"
            "\n{}".format(
        dt,
        total_time_taken,
        num_correct,
        expected_num_plates,
        num_incorrect,
        str_output
    ))
    f.close()
