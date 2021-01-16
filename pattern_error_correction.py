def correct_errors(plate_strings):
    # for string in plate_strings:
    plate_strings = simple_sequential_correction(plate_strings)
    return plate_strings


def simple_sequential_correction(plate_strings):
    MAX = 6  # minimum hamming distance between plates considered different

    for i in range(len(plate_strings) - 1):
        plate_strings[i] = str(hamming_distance(plate_strings[i], plate_strings[i + 1]))

    return plate_strings


def hamming_distance(str1, str2):
    dist = 0
    if len(str1) != len(str2):
        return "dist not equal"
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            dist += 1
    return dist
