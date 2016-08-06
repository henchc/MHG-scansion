# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Created at UC Berkeley 2015
# Authors: Christopher Hench
# ==============================================================================
'''This code trains and evaluates an n-gram tagger for MHG scansion based
on the paper presented at the NAACL-CLFL 2016 by Christopher Hench and
Alex Estes.'''

import codecs
import itertools
from itertools import chain
import nltk
import nltk.tag
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import NgramTagger
from CLFL_mdf_classification import classification_report, confusion_matrix
from CLFL_mdf_classification import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import itertools
import pandas as pd
import re
import random
import numpy as np

# created at UC Berkeley 2015
# Authors: Christopher Hench


def ngram_tagger(tagged_sents):
    patterns = [
        (r'''(b|c|d|f|g|h|j|k|l|m|n||p|q|r|s|t|v|w|x|z)e
        (b|c|d|f|g|h|j|k|l|m|n||p|q|r|s|t|v|w|x|z)''',
         'MORA'),
        (r'.*(a|e|i|o|u|ä|î|ô|ü)(a|e|i|o|u|ä|î|ô|ü)', 'DOPPEL'),
        (r'.*', 'MORA_HAUPT')]               # default
    regex_tagger = nltk.RegexpTagger(patterns)

    tagger1 = UnigramTagger(tagged_sents, backoff=regex_tagger)
    # cutoff = 3, if necessary
    tagger2 = BigramTagger(tagged_sents, backoff=tagger1)
    tagger3 = TrigramTagger(tagged_sents, backoff=tagger2)

    return tagger3


with open("Data/CLFL_dev-data.txt", 'r', encoding='utf-8') as f:
    tagged = f.read()  # text must be clean

tagged = tagged.split('\n')

newlines = []
for line in tagged:
    nl = "BEGL/BEGL WBY/WBY " + line + " WBY/WBY ENDL/ENDL"
    newlines.append(nl)

ftuples = []
for line in newlines:
    news = [nltk.tag.str2tuple(t) for t in line.split()]
    if len(news) > 0:
        ftuples.append(news)


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

    tagger = ngram_tagger(train_lines)

    # get report
    def bio_classification_report(y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from
        github master) to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        labs = [class_indices[cls] for cls in tagset]

        return((precision_recall_fscore_support(y_true_combined,
                                                y_pred_combined,
                                                labels=labs,
                                                average=None,
                                                sample_weight=None)),
               (classification_report(
                   y_true_combined,
                   y_pred_combined,
                   labels=[class_indices[cls] for cls in tagset],
                   target_names=tagset,
               )), labs)

    take_out = ["BEGL", "ENDL", "WBY"]

    def y_test_f(tagged_sents):
        return [[tag for (word, tag) in line if tag not in take_out]
                for line in tagged_sents]  # list of all the tags

    def y_pred_f(tagger, corpus):
        # notice we first untag the sentence
        return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

    y_test = y_test_f(test_lines)
    y_pred = y_test_f(y_pred_f(tagger, test_lines))

    bioc = bio_classification_report(y_test, y_pred)

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

    # to add and average cross validation
    if i != 0:
        df_all = df_all.add(df, axis="labels", fill_value=0)
    else:
        df_all = df

    print("Fold " + str(i) + " complete.\n")


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
