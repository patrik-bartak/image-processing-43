dutch_formats = [
    "XX-99-XX",  # 0
    "XX-XX-99",  # 1
    "99-XX-XX",  # 2
    "99-XXX-9",  # 3
    "9-XXX-99",  # 4
    "XX-999-X",  # 5
    "X-999-XX",  # 6
    "XXX-99-X",  # 7
    "X-99-XXX",  # 8
    "9-XX-999",  # 9
    "999-XX-9"  # 10
]

char_set = {"B", "D", "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "R", "S", "T",
            "V", "X", "Z"}

num_set = {"0", "1", "2", "3", "4",
           "5", "6", "7", "8", "9"}


def verify_format(string, is_dutch):
    string = length_check_and_hyphen_correction(string)
    if string is not None and len(string) == 8 and is_dutch and not is_dutch_format(string):
        string = None
    return string


def length_check_and_hyphen_correction(string):
    str_len = len(string)
    if str_len < 8:  # Disqualify plates with a length lower than 8
        return None
    while string[0] == "-":  # If a plate starts with hyphens, remove them
        string = string[1: str_len]
        str_len -= 1
    while string[str_len - 1] == "-":  # If a plate ends with hyphens, remove them
        string = string[0: str_len - 1]
        str_len -= 1
    return string


def is_dutch_format(string):
    if string[0] in char_set:
        if string[1] in char_set:
            if string[2] in char_set and string[3] == "-" and string[4] in num_set and string[5] in num_set and string[6] == "-" and string[7] in char_set:
                return True  # 7
            elif string[2] == "-":
                if string[3] in char_set and string[4] in char_set and string[5] == "-" and string[6] in num_set and string[7] in num_set:
                    return True  # 1
                elif string[3] in num_set:
                    if string[4] in num_set:
                        if string[5] in num_set and string[6] == "-" and string[7] in char_set:
                            return True  # 5
                        elif string[5] == "-" and string[6] in char_set and string[7] in char_set:
                            return True  # 0
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        elif string[1] in num_set:
            return None
        else:
            if string[2] in num_set and string[3] in num_set:
                if string[4] in num_set and string[5] == "-" and string[6] in char_set and string[7] in char_set:
                    return True  # 6
                elif string[4] == "-" and string[5] in char_set and string[6] in char_set and string[7] in char_set:
                    return True  # 8
                else:
                    return False
            else:
                return False
    elif string[0] in num_set:
        if string[1] in char_set:
            return None
        elif string[1] in num_set:
            # 23 10
            if string[2] in num_set and string[3] == "-" and string[4] in char_set and string[5] in char_set and string[6] == "-" and string[7] in num_set:
                return True  # 10
            elif string[2] == "-":
                if string[3] in char_set and string[4] in char_set:
                    if string[5] in char_set and string[6] == "-" and string[7] in num_set:
                        return True  # 3
                    elif string[5] == "-" and string[6] in char_set and string[7] in char_set:
                        return True  # 2
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            if string[2] in char_set and string[3] in char_set:
                if string[4] in char_set and string[5] == "-" and string[6] in num_set and string[7] in num_set:
                    return True  # 4
                elif string[4] == "-" and string[5] in num_set and string[6] in num_set and string[7] in num_set:
                    return True  # 9
                else:
                    return False
            else:
                return False
    else:
        return False

