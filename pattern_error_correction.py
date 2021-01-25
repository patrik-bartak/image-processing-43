from collections import Counter

confusing_JI = {'J', 'I'}
confusing_MH = {'M', 'H'}


def correct_errors(plate_strings):
    # for string in plate_strings:
    plate_strings = simple_sequential_correction(plate_strings)
    return plate_strings


def simple_sequential_correction(plate_strings):
    if len(plate_strings) == 0:
        return None

    min_diff = 3  # minimum hamming distance between plates considered different

    index_list = [0]
    for i in range(len(plate_strings) - 1):
        diff = hamming_distance(plate_strings[i], plate_strings[i + 1])
        if diff is None:
            continue
        if diff >= min_diff:
            # plate_strings[i] += " - NEW PLATE NEXT"
            index_list.append(i + 1)
    index_list.append(len(plate_strings))

    most_common = []
    for i in range(len(index_list) - 1):
        # https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
        # append the most common item out of the substrings defined by the index_list
        most_common.append(Counter(plate_strings[index_list[i]: index_list[i + 1]]).most_common(1)[0][0])

    return most_common


def hamming_distance(str1, str2):
    dist = 0
    if len(str1) != len(str2):
        return None
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            dist += 1
    return dist
