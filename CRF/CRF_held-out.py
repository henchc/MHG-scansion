# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# Created at UC Berkeley 2015
# Authors: Christopher Hench
# ==============================================================================
'''This code trains and evaluates a CRF model for MHG scansion based
on the paper presented at the NAACL-CLFL 2016 by Christopher Hench and
Alex Estes. This model is for the held-out data.'''

import codecs
from pickle import dump
from CLFL_mdf_classification import confusion_matrix
import pycrfsuite
import numpy as np
import pandas as pd
from scan_text_rev import only_four_stresses
from process_feats import syllable2features, line2features, line2labels, line2tokens
from prep_crf import prep_crf
from new_bio_class_report import bio_classification_report


# open hand-tagged data
with open("Data/CLFL_dev-data.txt", 'r', encoding='utf-8') as f:
    training_tagged = f.read()

# add features to data
ftuples = prep_crf(training_tagged)

# open hand-tagged data
with open("Data/CLFL_held-out.txt", 'r', encoding='utf-8') as f:
    heldout = f.read()

# add features to data
htuples = prep_crf(heldout)
sylls_list = [[t[0] for t in l] for l in htuples]


test_lines = htuples
train_lines = ftuples

X_train = [line2features(s) for s in train_lines]
y_train = [line2labels(s) for s in train_lines]

X_test = [line2features(s) for s in test_lines]
y_test = [line2labels(s) for s in test_lines]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# parameters to fiddle with
trainer.set_params({
    'c1': 1.3,   # coefficient for L1 penalty
    'c2': 10e-4,  # coefficient for L2 penalty
    'num_memories': 6,  # default is 6
    # 'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': False,
    # 'max_linesearch': 1000,
    # 'linesearch': 'Backtracking'
    # 'feature.minfreq': 5
    # 'feature.possible_states': True,
})


# run trainer and tagger
trainer.params()
trainer.train('MHGMETRICS_heldout.crfsuite')
tagger = pycrfsuite.Tagger()
tagger.open('MHGMETRICS_heldout.crfsuite')


# y_pred = [tagger.tag(xseq) for xseq in X_test]
y_pred = only_four_stresses(X_test, tagger, sylls_list)
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
    "MORA",
    "MORA_HAUPT"]
abs_labels = [l for l in ext_labels if l not in all_labels]

# print(bio_classification_report(y_test, y_pred)[1])

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


# TO PRINT HELPFUL/NOT HELPFUL RULES
from collections import Counter
info = tagger.info()


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])
