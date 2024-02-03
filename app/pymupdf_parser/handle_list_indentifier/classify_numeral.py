"""Copyright (C) 2022 Adarsh Gupta"""
import string

lower_case_set = {s: i + 1 for i, s in enumerate(string.ascii_lowercase)}
upper_case_set = {s: i + 1 for i, s in enumerate(string.ascii_uppercase)}

number_set = {f"{i}": i for i in range(1000)}

lower_roman_set = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}
upper_roman_set = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}

numeral_dict = {
    "lower_case": lower_case_set,
    "upper_case": upper_case_set,
    "number": number_set,
    "lower_roman": lower_roman_set,
    "upper_roman": upper_roman_set,
}


def classify_numeral(key):
    classes = []
    numeral_value = []
    for numeral_key, number_set in numeral_dict.items():
        if key in number_set:
            classes.append(numeral_key)
            numeral_value.append(number_set[key])

    if not classes:
        classes.append(key)
        numeral_value.append(0)

    return classes, numeral_value
