'''Train a Bidirectional LSTM for MHG scansion.'''

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.core import Masking
from keras.layers import TimeDistributed, Dense
from keras.layers import Dropout, Embedding, LSTM, Input, merge
from prep_nn import prep_scan
from keras.utils import np_utils, generic_utils


np.random.seed(1337)  # for reproducibility
nb_words = 20000  # max. size of vocab
nb_classes = 10  # number of labels
batch_size = 5  # create and update net after 10 lines

# input for X is multi-dimensional numpy array with syll IDs,
# one line per array. input y is multi-dimensional numpy array with
# binary arrays for each value of each label.
# maxlen is length of longest line
print('Loading data...')
(X_train, y_train), (X_test, y_test), maxlen, sylls_ids, tags_ids = prep_scan(
    nb_words=nb_words, test_split=0.1)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(maxlen,), dtype='int32')

# this embedding layer will transform the sequences of integers
# into vectors of size 256
embedded = Embedding(nb_words, output_dim=256, input_length=maxlen)(sequence)

# apply forwards LSTM
forwards = LSTM(output_dim=256, return_sequences=True)(embedded)
# apply backwards LSTM
backwards = LSTM(output_dim=256, return_sequences=True,
                 go_backwards=True)(embedded)

# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)

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
          nb_epoch=5,
          shuffle=True,
          validation_data=[X_test, y_test])

# predict
X_test_new = X_test[:3]

# first get probabilities of labels in binary arrays, then convert to classes
predicted_arrays = model.predict(X_test_new, batch_size=batch_size)


# return list of labels for input sequence
def arrays_to_labels(y_pred_arrays):
    return [[np.argmax(arr) for arr in seq] for seq in y_pred_arrays]

predicted_labels = arrays_to_labels(predicted_arrays)

# print actual sylls and labels index from ID dicts
tagged_lines = []
for i, line in enumerate(X_test_new):
    tagged_line = []
    for i2, v in enumerate(line):
        if v != 0:
            tagged_line.append(
                (sylls_ids[v], tags_ids[predicted_labels[i][i2]]))
    tagged_lines.append(tagged_line)

print()
for line in tagged_lines:
    print(line)
