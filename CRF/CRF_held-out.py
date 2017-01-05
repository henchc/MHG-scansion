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
import nltk
import nltk.tag
from pickle import dump
import itertools
from itertools import chain
from CLFL_mdf_classification import classification_report, confusion_matrix
from CLFL_mdf_classification import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import re
import numpy as np
import pandas as pd
from scan_text_rev import only_four_stresses


def syllableend(syl):
    vowels = 'aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ'

    # word and line boundaries don't matter
    if syl == 'BEGL':
        ending = "NAB"
    elif syl == 'ENDL':
        ending = "NAE"
    elif syl == 'WBY':
        ending = 'NAW'

    # open syllables
    elif len(syl) > 1 and syl[-1] in vowels:
        ending = "O"

    # account for syllables of one letter
    elif len(syl) == 1:
        if str(syl) in vowels:
            ending = "O"
        else:
            ending = "C"

    # close syllables
    else:
        ending = "C"

    return(ending)


def syllableweight(syl):
    vowels = 'aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ'
    longvowels = "âæāêēîīôœōûū"

    # ending not a vowel, heavy
    if len(syl) > 1 and syl[-1] not in vowels:
        weight = "H"

    # ending double vowel, heavy
    elif len(syl) > 1 and syl[-2] in vowels and syl[-1] in vowels:
        weight = "H"

    # ending long vowel, heavy
    elif len(syl) > 1 and syl[-1] in longvowels:
        weight = "H"

    elif len(syl) == 1:
        if str(syl) in longvowels:
            weight = "H"
        else:
            weight = "L"

    else:
        weight = "L"

    return(weight)


def prepcrf(taggedstring):

    taggedstring = taggedstring.replace("MORA_NEBEN", "MORA_HAUPT")
    taggedstring = taggedstring.replace("HALB_NEBEN", "HALB_HAUPT")

    # break data into list on line breaks
    tagged = taggedstring.split('\n')

    # create tuples with syllable characteristic features
    training = []
    for line in tagged:
        news = [nltk.tag.str2tuple(t)
                for t in line.split()]  # make tags into tuples
        if len(news) > 0:
            newtups = []
            for tup in news:
                # adds syllable characteristics to tuple for being feature
                syllweight = (
                    tup[0], syllableend(
                        tup[0]), syllableweight(
                        tup[0]), tup[1])
                newtups.append(syllweight)
            training.append(newtups)

    # add features for word boundary and line boundary
    ftuples = []
    for line in training:
        ltuples = []
        for i, tup in enumerate(line):
            if tup[0] == 'WBY':
                pass
            elif i == 0:
                lpos = "BEGL"
                if line[i + 1][0] == "WBY":
                    wpos = "MONO"
                else:
                    wpos = "WBYL"
                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

            elif i > 0 and i < len(line) - 1:

                lpos = "LPNA"

                if line[i - 1][0] == "WBY" and line[i + 1][0] == "WBY":
                    wpos = "MONO"
                elif line[i - 1][0] == "WBY":
                    wpos = "WBYL"
                elif line[i + 1][0] == "WBY":
                    wpos = "WBYR"
                else:
                    wpos = "WBNA"

                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

            elif i == len(line) - 1:
                lpos = "ENDL"
                if line[i - 1][0] == "WBY":
                    wpos = "MONO"
                else:
                    wpos = "WBYR"
                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

        ftuples.append(ltuples)

    return (ftuples)


# open hand-tagged data
with open("Data/CLFL_dev-data.txt", 'r', encoding='utf-8') as f:
    training_tagged = f.read()

# add features to data
ftuples = prepcrf(training_tagged)

# open hand-tagged data
with open("Data/CLFL_held-out.txt", 'r', encoding='utf-8') as f:
    heldout = f.read()

# add features to data
htuples = prepcrf(heldout)
sylls_list = [[t[0] for t in l] for l in htuples]


# designate features for model to collect
def syllable2features(line, i):
    syllable = line[i][0]
    oc = line[i][1]  # light, heavy, open, closed
    lh = line[i][2]
    lpos = line[i][3]  # position in line
    wpos = line[i][4]  # position in word

    features = [
        'bias',  # bias to current syllable

        'len(syllable)=' + str(len(syllable)),  # length of syllable

        'syllable[:1]=' + syllable[:1],  # first letter
        'syllable[:2]=' + syllable[:2],  # first two letters
        'syllable[-1:]=' + syllable[-1:],  # last letter
        'syllable[-2:]=' + syllable[-2:],  # last two letters


        'oc=' + oc,
        'lh=' + lh,
        'lpos=' + lpos,
        'wpos=' + wpos,

    ]

    for p in range(1, len(line)):
        if i > p:
            syllablen1 = line[i - p][0]
            ocn1 = line[i - p][1]
            lhn1 = line[i - p][2]
            lposn1 = line[i - p][3]
            wposn1 = line[i - p][4]
            features.extend([
                '-' + str(p) + ':len(syllable)=' + str(len(syllablen1)),

                '-' + str(p) + ':syllable[:1]=' + syllablen1[:1],
                '-' + str(p) + ':syllable[:2]=' + syllablen1[:2],
                '-' + str(p) + ':syllable[-1:]=' + syllablen1[-1:],
                '-' + str(p) + ':syllable[-2:]=' + syllablen1[-2:],

                '(' + str(p) + ':syllable[-1] + syllable[0])=%s' +
                (syllablen1[-2:] + syllable[:2]),

                '-' + str(p) + ':oc=' + ocn1,
                '-' + str(p) + ':lh=' + lhn1,
                '-' + str(p) + ':lpos=' + lposn1,
                '-' + str(p) + ':wpos=' + wposn1,
            ])

        else:
            features.append('BOL')

    for p in range(1, len(line)):
        if i < len(line) - p:
            syllable1 = line[i + p][0]
            oc1 = line[i + p][1]
            lh1 = line[i + p][2]
            lpos1 = line[i + p][3]
            wpos1 = line[i + p][4]
            features.extend([
                '+' + str(p) + ':len(syllable)=' + str(len(syllable1)),

                '+' + str(p) + ':syllable[:1]=' + syllable1[:1],
                '+' + str(p) + ':syllable[:2]=' + syllable1[:2],
                '+' + str(p) + ':syllable[-1:]=' + syllable1[-1:],
                '+' + str(p) + ':syllable[-2:]=' + syllable1[-2:],

                '(-' + str(p) + ':syllable[-1] + syllable[0])=%s' +
                (syllable[-2:] + syllable1[:2]),

                '+' + str(p) + ':oc=' + oc1,
                '+' + str(p) + ':lh=' + lh1,
                '+' + str(p) + ':lpos=' + lpos1,
                '+' + str(p) + ':wpos=' + wpos1,

            ])

        else:
            features.append('EOL')

    return features


def line2features(line):
    return [syllable2features(line, i) for i in range(len(line))]


def line2labels(line):
    return [label for token, oc, lh, lpos1, wpos1, label in line]


def line2tokens(line):
    return [token for token, oc, lh, lpos1, wpos1, label in line]


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
    # 'num_memories': 10,
    # 'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True,
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


# TO PRINT STATS

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


# # TO PRINT HELPFUL/NOT HELPFUL RULES
# from collections import Counter
# info = tagger.info()

# def print_transitions(trans_features):
#     for (label_from, label_to), weight in trans_features:
#         print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

# print("Top likely transitions:")
# print_transitions(Counter(info.transitions).most_common(15))

# print("\nTop unlikely transitions:")
# print_transitions(Counter(info.transitions).most_common()[-15:])

# def print_state_features(state_features):
#     for (attr, label), weight in state_features:
#         print("%0.6f %-6s %s" % (weight, label, attr))

# print("Top positive:")
# print_state_features(Counter(info.state_features).most_common(20))

# print("\nTop negative:")
# print_state_features(Counter(info.state_features).most_common()[-20:])
