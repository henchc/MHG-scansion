'''This function preps multi-classfication data inputs for keras.
Data is annotated according to the str2tuple method in NLTK.'''

# Authors: Christopher Hench
# ==============================================================================

from __future__ import absolute_import
import numpy as np
from keras.utils import np_utils, generic_utils
from keras.preprocessing import sequence


def prep_scan(nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    from nltk import str2tuple

    with open("Data/CLFL_all_data.txt", "r") as f:
        raw_data = f.read()

    # separate sylls and labels and reject WBY
    data = [str2tuple(x) for x in raw_data.split()]
    data_lines = [[str2tuple(x) for x in line.split()]
                  for line in raw_data.split('\n')]
    data_lines = [[tup for tup in line if tup[0] != "WBY"] for line in
                  data_lines]

    # sylls to IDs
    sylls = [x[0] for x in data]
    sylls_lines = [[x[0] for x in line] for line in data_lines]
    sylls_set = list(set(sylls))
    sylls_ids = {}
    rev_sylls_ids = {}
    for i, x in enumerate(sylls_set):
        sylls_ids[x] = i + 1  # so we can pad with 0s
        rev_sylls_ids[i + 1] = x

    # labels to IDs
    tags = [x[1] for x in data]
    tags_lines = [[x[1] for x in line] for line in data_lines]
    tags_set = list(set(tags))
    print(len(tags_set))
    tags_ids = {}
    rev_tags_ids = {}
    for i, x in enumerate(tags_set):
        tags_ids[x] = i + 1  # so we can pad with 0s
        rev_tags_ids[i + 1] = x

    # lines of syll IDs
    all_sylls_ids = []
    for line in sylls_lines:
        s_l = [sylls_ids[x] for x in line]
        all_sylls_ids.append(s_l)

    # lines of label IDs
    all_tags_ids = []
    for line in tags_lines:
        t_l = [tags_ids[x] for x in line]
        all_tags_ids.append(t_l)

    X, labels = all_sylls_ids, all_tags_ids
    maxlen = len(max(labels, key=len))  # longest line in items

    # train and test split
    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    # pad with 0s
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, value=0.)  # must be float
    X_test = sequence.pad_sequences(X_test, value=0.)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # need to pad y too, because more than 1 output value
    y_train = sequence.pad_sequences(np.array(y_train), value=0.)
    y_test = sequence.pad_sequences(np.array(y_test), value=0.)

    y_train = [np_utils.to_categorical(y) for y in y_train]
    y_test = [np_utils.to_categorical(y) for y in y_test]

    # create 3D array for keras multi-classfication
    new_y_train = []
    for array in y_train:
        if len(array[0]) < 10:
            to_add = 10 - len(array[0])
            new_y_train.append(np.hstack((array, np.zeros((array.shape[0],
                                                           to_add)))))
        else:
            new_y_train.append(array)

    y_train = np.asarray(new_y_train)

    # create 3D array for keras multi-classfication
    new_y_test = []
    for array in y_test:
        if len(array[0]) < 10:
            to_add = 10 - len(array[0])
            new_y_test.append(np.hstack((array, np.zeros((array.shape[0],
                                                          to_add)))))
        else:
            new_y_test.append(array)

    y_test = np.asarray(new_y_test)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    return ((X_train, y_train), (X_test, y_test), maxlen, rev_sylls_ids,
            rev_tags_ids)

if __name__ == "__main__":
    prep_scan()
