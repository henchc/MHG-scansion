# -*- coding: utf-8 -*-
from __future__ import unicode_literals  # for python2 compatibility
from __future__ import division
from __future__ import absolute_import

# created at UC Berkeley 2015
# Authors: Christopher Hench

# This program scans MHG epic poetry, returning data to analyze statistically

import codecs
import pycrfsuite
from get_features import get_features
from meteryield import mhgscansion
from syllable2features import syllable2features


def line2features(line):
    return [syllable2features(line, i) for i in range(len(line))]


tagger = pycrfsuite.Tagger()
tagger.open('MHG_METER.crfsuite')


with open("parzival.txt", 'r', encoding='utf-8') as f:
    text = f.read()  # text must be clean

text_with_features = get_features(text)

text_tags = [tagger.tag(line2features(line)) for line in text_with_features]

# add back tags to features
features_and_tags = []
for i, line in enumerate(text_with_features):
    line_features_and_tags = []
    for i2, syll in enumerate(line):
        line_features_and_tags.append(syll + (text_tags[i][i2],))
    features_and_tags.append(line_features_and_tags)

# return words, sylls and labels
words_sylls_labels = []
for line in features_and_tags:
    line_words = []
    rec_word = []
    for syll in line:
        if syll[3] == "WBYR":
            rec_word.append((syll[0], syll[-1]))
            line_words.append(rec_word)
            rec_word = []
        elif syll[3] == "MONO":
            line_words.append([(syll[0], syll[-1])])
            rec_word = []
        else:
            rec_word.append((syll[0], syll[-1]))

    words_sylls_labels.append(line_words)

# change primary stresses to secondary where necessary
tags_n_stress = []
for line in words_sylls_labels:
    rev_line = []
    for word in line:
        rev_word = []
        stress_present = 0
        for syll in word:
            rev_syll = syll
            if syll[-1] == "MORA_HAUPT" or syll[-1] == "DOPPEL" or syll[-1] == "HALB_HAUPT":
                stress_present += 1
                if stress_present > 1:
                    if syll[-1] == "MORA_HAUPT":
                        rev_syll = (syll[0], "MORA_NEBEN")
                    elif syll[-1] == "HALB_HAUPT":
                        rev_syll = (syll[0], "HALB_NEBEN")

            rev_word.append(rev_syll)
        rev_line.append(rev_word)
    tags_n_stress.append(rev_line)

print(tags_n_stress)

# make nice symbols (not necessary for data analysis, only for print)
# towrite = mhgscansion(tags_n_stress)
#
# with codecs.open('scanned_text.txt', 'w', 'utf-8') as f:
#     f.write(to_write)
