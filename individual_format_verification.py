dutch_formats = {
    "X": {
        "X": {
            "X": "XXX-99-X",
            "-": {
                "X": "XX-XX-99",
                "9": {
                    "9": {
                        "9": "XX-999-X",
                        "-": "XX-99-XX"
                    }
                }
            }
        },
        "-": "X-999-XX"
    },
    "9": {
        "9": {
            "9": "999-XX-9",
            "-": {
                "X": {
                    "X": {
                        "X": "99-XXX-9",
                        "-": "99-XX-XX"
                    }
                }
            }
        },
        "-": {
            "X": {
                "X": {
                    "X": "9-XXX-99",
                    "-": "9-XX-999"
                }
            }
        }
    }
}

char_set = {"B", "D", "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "R", "S", "T",
            "V", "X", "Z"}

num_set = {"0", "1", "2", "3", "4",
           "5", "6", "7", "8", "9"}


def verify_format(string, is_dutch):
    string = length_check_and_hyphen_correction(string)
    if string is None or len(string) != 8 or not is_dutch_format(string):
        string = None
    return string


def length_check_and_hyphen_correction(string):
    str_len = len(string)
    while len(string) > 1 and string[0] == "-":  # If a plate starts with hyphens, remove them
        string = string[1: str_len]
        str_len -= 1
    while len(string) > 1 and string[str_len - 1] == "-":  # If a plate ends with hyphens, remove them
        string = string[0: str_len - 1]
        str_len -= 1
    if str_len < 8:  # Disqualify plates with a length lower than 8
        return None
    return string


def is_dutch_format(string):
    return dutch_helper(string, dutch_formats, 0)


def dutch_helper(string, dictionary, index):
    if index >= len(string):
        return True

    if isinstance(dictionary, str):
        for i in range(index, len(string)):
            temp = string[i]
            if temp in char_set:
                temp = "X"
            if temp in num_set:
                temp = "9"
            if temp != dictionary[i]:
                return False
        return True

    temp = string[index]
    if temp in char_set:
        temp = "X"
    if temp in num_set:
        temp = "9"

    if temp not in dictionary:
        return False

    return dutch_helper(string, dictionary[temp], index + 1)
