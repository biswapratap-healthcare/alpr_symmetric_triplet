import string

empty_encodings = [0 for _ in range(0, 37, 1)]
one_hot_map = dict()
for idx in range(0, 10, 1):
    encoding = empty_encodings[:]
    encoding[idx] = 1
    one_hot_map[str(idx)] = encoding
alphabet_string = string.ascii_uppercase
alphabet_list = list(alphabet_string)
for idx, alphabet in enumerate(alphabet_list):
    encoding = empty_encodings[:]
    idx = idx + 10
    encoding[idx] = 1
    one_hot_map[alphabet] = encoding
encoding = empty_encodings[:]
encoding[-1] = 1
one_hot_map['#'] = encoding
