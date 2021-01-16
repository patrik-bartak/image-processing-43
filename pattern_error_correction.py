def correct_errors(plate_strings):
    # for string in plate_strings:
    plate_strings = simple_sequential_correction(plate_strings)
    return plate_strings


def simple_sequential_correction(plate_strings):
    min_diff = 6  # minimum hamming distance between plates considered different

    for i in range(len(plate_strings) - 1):
        dist = hamming_distance(plate_strings[i], plate_strings[i + 1])
        if dist is None:
            continue
        if dist >= min_diff:
            plate_strings[i] += " - NEW PLATE NEXT"

    return plate_strings


def hamming_distance(str1, str2):
    dist = 0
    if len(str1) != len(str2):
        return None
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            dist += 1
    return dist
