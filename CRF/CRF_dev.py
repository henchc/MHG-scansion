# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

# Created at UC Berkeley 2015
# Authors: Christopher Hench
# ==============================================================================
'''This code trains and evaluates a CRF model for MHG scansion based
on the paper presented at the NAACL-CLFL 2016 by Christopher Hench and
Alex Estes. This model is for tuning.'''

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
import random

# open hand-tagged data
with open("Data/CLFL_dev-data.txt", 'r', encoding='utf-8') as f:
    tagged = f.read()

# add features to data
ftuples = prep_crf(tagged)

# for 10 fold validation
num_folds = 10
subset_size = int(len(ftuples) / num_folds)
rand_all = random.sample(range(0, len(ftuples)), len(ftuples))
test_inds = [rand_all[x:x + subset_size]
             for x in range(0, len(rand_all), subset_size)]


for i, inds in enumerate(test_inds):

    test_inds = inds
    train_inds = list(set(range(0, len(ftuples))) - set(test_inds))

    test_lines = []
    train_lines = []

    for x in test_inds:
        test_lines.append(ftuples[x])

    for x in train_inds:
        train_lines.append(ftuples[x])

    sylls_list = [[t[0] for t in l] for l in test_lines]

    # separate data and train
    X_train = [line2features(s) for s in train_lines]
    y_train = [line2labels(s) for s in train_lines]

    X_test = [line2features(s) for s in test_lines]
    y_test = [line2labels(s) for s in test_lines]

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # parameters to fiddle with
    trainer.set_params({
        'c1': 1.3,   # coefficient for L1 penalty default 0
        'c2': 10e-4,  # coefficient for L2 penalty default 1 or -4
        'num_memories': 6,  # default is 6
        # 'delta': 1e-5,  # 1e-5 is default
        # 'max_iterations': 100,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': False,  # default is False
        # 'max_linesearch': 1000,  # default 20
        # 'linesearch': 'Backtracking'  # default MoreThuente
        # 'feature.minfreq': 5
        # 'feature.possible_states': True,  # default is False

        # # averaged perceptron
        # 'max_iterations' : 1000  # default is 100
    })

    # run trainer and tagger
    trainer.params()

    '''
    default is lbfgs
    • ‘lbfgs’ for Gradient descent using the L-BFGS method,
    • ‘l2sgd’ for Stochastic Gradient Descent with L2 regularization term
    • ‘ap’ for Averaged Perceptron
    • ‘pa’ for Passive Aggressive
    • ‘arow’ for Adaptive Regularization Of Weight Vector
    '''

    # trainer.select('ap', type='crf1d')
    trainer.train('MHGMETRICS_dev.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('MHGMETRICS_dev.crfsuite')

    # get report

    # y_pred = [tagger.tag(xseq) for xseq in X_test]
    y_pred = only_four_stresses(X_test, tagger, sylls_list)

    bioc = bio_classification_report(y_test, y_pred)

    print(bioc[1])

    # to parse
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

    ext_labels = ["DOPPEL", "EL", "HALB", "HALB_HAUPT", "MORA", "MORA_HAUPT"]
    abs_labels = [l for l in ext_labels if l not in all_labels]

    # print(bio_classification_report(y_test, y_pred)[1])

    data = {"labels": all_labels, "precision": p, "recall": r,
            "f1": f1, "support": s, "tots": tot_avgs, "all_s": all_s}

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

    # to add and average cross validation
    if i != 0:
        df_all = df_all.add(df, axis="labels", fill_value=0)
    else:
        df_all = df

    # # TO PRINT HELPFUL/NOT HELPFUL RULES
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

    print("Fold " + str(i + 1) + " complete.\n")


df_all["p_AVG"] = df_all.w_p / df_all.support
df_all["r_AVG"] = df_all.w_r / df_all.support
df_all["f1_AVG"] = df_all.w_f1 / df_all.support
df_all["tots_AVG"] = df_all.w_tots / df_all.all_s

df_all = df_all.drop("f1", 1)
df_all = df_all.drop("precision", 1)
df_all = df_all.drop("recall", 1)
df_all = df_all.drop("tots", 1)
df_all = df_all.drop("w_p", 1)
df_all = df_all.drop("w_r", 1)
df_all = df_all.drop("w_f1", 1)
df_all = df_all.drop("w_tots", 1)


print(df_all)
