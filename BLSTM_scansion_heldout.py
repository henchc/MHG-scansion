'''Train a Bidirectional LSTM for MHG scansion.
Current accuracy 91.88 on validation data'''
# Authors: Christopher Hench
# ==============================================================================

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.core import Masking
from keras.layers import TimeDistributed, Dense
from keras.layers import Dropout, Embedding, LSTM, Input, merge
from prep_nn import prep_scan
from keras.utils import np_utils, generic_utils
import itertools
from itertools import chain
from CLFL_mdf_classification import classification_report, confusion_matrix
from CLFL_mdf_classification import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pandas as pd


np.random.seed(1337)  # for reproducibility
nb_words = 20000  # max. size of vocab
nb_classes = 10  # number of labels
# hidden = int(((953 + 10) / 2))
hidden = 500
batch_size = 10  # create and update net after 10 lines
val_split = .1
epochs = 6

# input for X is multi-dimensional numpy array with syll IDs,
# one line per array. input y is multi-dimensional numpy array with
# binary arrays for each value of each label.
# maxlen is length of longest line
print('Loading data...')
(X_train, y_train), (X_test, y_test), maxlen, sylls_ids, tags_ids = prep_scan(
    nb_words=nb_words, test_len=75)

print(len(X_train), 'train sequences')
print(int(len(X_train)*val_split), 'validation sequences')
print(len(X_test), 'heldout sequences')

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(maxlen,), dtype='int32')

# this embedding layer will transform the sequences of integers
# into vectors of size 256
embedded = Embedding(nb_words, output_dim=hidden,
                     input_length=maxlen, mask_zero=True)(sequence)

# apply forwards LSTM
forwards = LSTM(output_dim=hidden, return_sequences=True)(embedded)
# apply backwards LSTM
backwards = LSTM(output_dim=hidden, return_sequences=True,
                 go_backwards=True)(embedded)

# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.15)(merged)

# TimeDistributed for sequence
# change activation to sigmoid?
output = TimeDistributed(
    Dense(output_dim=nb_classes,
          activation='softmax'))(after_dp)

model = Model(input=sequence, output=output)

# try using different optimizers and different optimizer configs
# loss=binary_crossentropy, optimizer=rmsprop
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=epochs,
          shuffle=True,
          validation_split=val_split,
          sample_weight=0.)


# held-out testing:
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    labs = [class_indices[cls] for cls in tagset]

    return((precision_recall_fscore_support(y_true_combined, y_pred_combined,
                                            labels=labs,
                                            average=None,
                                            sample_weight=None)),
           (classification_report(
               y_true_combined,
               y_pred_combined,
               labels=[class_indices[cls] for cls in tagset],
               target_names=tagset,
           )), labs)


# first get probabilities of labels in binary arrays, then convert to classes
predicted_arrays = model.predict(X_test, batch_size=batch_size)


# return list of labels for input sequence of binary arrays
def arrays_to_labels(y_pred_arrays):
    return [[np.argmax(arr) for arr in seq] for seq in y_pred_arrays]

pred_classes = arrays_to_labels(predicted_arrays)
y_test_classes = arrays_to_labels(y_test)

# get list of lines of labels
y_pred = []
for i, line in enumerate(pred_classes):
    line_labels = []
    for v in line:
        if v != 0:
            line_labels.append(tags_ids[v])
    y_pred.append(line_labels)

y_test = []
for i, line in enumerate(y_test_classes):
    line_labels = []
    for v in line:
        if v != 0:
            line_labels.append(tags_ids[v])
    y_test.append(line_labels)

# get stats
bioc = bio_classification_report(y_test, y_pred)

p, r, f1, s = bioc[0]

tot_avgs = []

for v in (np.average(p, weights=s),
          np.average(r, weights=s),
          np.average(f1, weights=s)):
    tot_avgs.append(v)

toext = [0] * (len(s) - 3)
tot_avgs.extend(toext)

all_s = [sum(s)] * len(s)

rep = bioc[1]
all_labels = []

for word in rep.split():
    if word.isupper():
        all_labels.append(word)

ext_labels = [
    "DOPPEL",
    "EL",
    "HALB",
    "HALB_HAUPT",
    "HALB_NEBEN",
    "MORA",
    "MORA_HAUPT",
    "MORA_NEBEN"]
abs_labels = [l for l in ext_labels if l not in all_labels]

data = {
    "labels": all_labels,
    "precision": p,
    "recall": r,
    "f1": f1,
    "support": s,
    "tots": tot_avgs,
    "all_s": all_s}

df = pd.DataFrame(data)

if len(abs_labels) > 0:
    if "HALB_NEBEN" in abs_labels:
        line = pd.DataFrame({"labels": "HALB_NEBEN",
                             "precision": 0,
                             "recall": 0,
                             "f1": 0,
                             "support": 0,
                             "tots": 0,
                             "all_s": 0},
                            index=[4])
        df = pd.concat([df.ix[:3], line, df.ix[4:]]).reset_index(drop=True)
    if "EL" in abs_labels:
        line = pd.DataFrame({"labels": "EL",
                             "precision": 0,
                             "recall": 0,
                             "f1": 0,
                             "support": 0,
                             "tots": 0,
                             "all_s": 0},
                            index=[1])
        df = pd.concat([df.ix[0], line, df.ix[1:]]).reset_index(drop=True)

df["w_p"] = df.precision * df.support
df["w_r"] = df.recall * df.support
df["w_f1"] = df.f1 * df.support
df["w_tots"] = df.tots * df.all_s
df_all = df

print(df_all)
