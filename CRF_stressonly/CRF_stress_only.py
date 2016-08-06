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
import random


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

    taggedstring = taggedstring.replace("MORA_HAUPT", "HEBUNG")
    taggedstring = taggedstring.replace("MORA_NEBEN", "HEBUNG")
    taggedstring = taggedstring.replace("MORA", "SENKUNG")
    taggedstring = taggedstring.replace("HALB_HAUPT", "HEBUNG")
    taggedstring = taggedstring.replace("HALB_NEBEN", "HEBUNG")
    taggedstring = taggedstring.replace("HALB", "SENKUNG")
    taggedstring = taggedstring.replace("DOPPEL", "HEBUNG")
    taggedstring = taggedstring.replace("EL", "SENKUNG")

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
                syllweight = (tup[0], syllableend(tup[0]),
                              syllableweight(tup[0]), tup[1])
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
    tagged = f.read()

# add features to data
ftuples = prepcrf(tagged)

# designate features for model to collect


def syllable2features(line, i):
    syllable = line[i][0]
    lh = line[i][1]  # light, heavy, open, closed
    oc = line[i][2]
    lpos = line[i][3]  # position in line
    wpos = line[i][4]  # position in word

    features = [
        'bias',  # bias to current syllable
        'i=' + str(i),  # position in line

        'len(syllable)=' + str(len(syllable)),  # length of syllable

        'syllable[:1]=' + syllable[:1],  # first letter
        'syllable[:2]=' + syllable[:2],  # first two letters
        'syllable[-1:]=' + syllable[-1:],  # last letter
        'syllable[-2:]=' + syllable[-2:],  # last two letters
        'syllable[-3:]=' + syllable[-3:],  # last threel letters


        'lh=' + lh,
        'oc=' + oc,
        'lpos=' + lpos,
        'wpos=' + wpos,

    ]
    if i > 0:  # preceding syllable
        syllablen1 = line[i - 1][0]
        lhn1 = line[i - 1][1]
        ocn1 = line[i - 1][2]
        lposn1 = line[i - 1][3]
        wposn1 = line[i - 1][4]
        features.extend([
            '-1:i=' + str(i - 1),

            '-1:len(syllable)=' + str(len(syllablen1)),

            '-1:syllable[:1]=' + syllablen1[:1],
            '-1:syllable[:2]=' + syllablen1[:2],
            '-1:syllable[-1:]=' + syllablen1[-1:],
            '-1:syllable[-2:]=' + syllablen1[-2:],
            '-1:syllable[-3:]=' + syllablen1[-3:],

            '(-1:syllable[-1] + syllable[0])=%s' +
            (syllablen1[-2:] + syllable[:2]),

            '-1:lh=' + lhn1,
            '-1:oc=' + ocn1,
            '-1:lpos=' + lposn1,
            '-1:wpos=' + wposn1,
        ])
    else:
        features.append('BOL')

    if i > 1:
        syllablen2 = line[i - 2][0]
        lhn2 = line[i - 2][1]
        ocn2 = line[i - 2][2]
        lposn2 = line[i - 2][3]
        wposn2 = line[i - 2][4]
        features.extend([
            '-2:i=' + str(i - 2),

            '-2:len(syllable)=' + str(len(syllablen2)),

            '-2:syllable[:1]=' + syllablen2[:1],
            '-2:syllable[:2]=' + syllablen2[:2],
            '-2:syllable[-1:]=' + syllablen2[-1:],
            '-2:syllable[-2:]=' + syllablen2[-2:],
            '-2:syllable[-3:]=' + syllablen2[-3:],

            '-2:lh=' + lhn2,
            '-2:oc=' + ocn2,
            '-2:lpos=' + lposn2,
            '-2:wpos=' + wposn2,

        ])
    else:
        features.append('BOL')

    if i > 2:
        syllablen3 = line[i - 3][0]
        lhn3 = line[i - 3][1]
        ocn3 = line[i - 3][2]
        lposn3 = line[i - 3][3]
        wposn3 = line[i - 3][4]
        features.extend([
            '-3:i=' + str(i - 3),

            '-3:len(syllable)=' + str(len(syllablen3)),

            '-3:syllable[:1]=' + syllablen3[:1],
            '-3:syllable[:2]=' + syllablen3[:2],
            '-3:syllable[-1:]=' + syllablen3[-1:],
            '-3:syllable[-2:]=' + syllablen3[-2:],
            '-3:syllable[-3:]=' + syllablen3[-3:],

            '-3:lh=' + lhn3,
            '-3:oc=' + ocn3,
            '-3:lpos=' + lposn3,
            '-3:wpos=' + wposn3,

        ])
    else:
        features.append('BOL')

    if i > 3:
        syllablen4 = line[i - 4][0]
        lhn4 = line[i - 4][1]
        ocn4 = line[i - 4] [2]
        lposn4 = line[i - 4][3]
        wposn4 = line[i - 4][4]
        features.extend([
            '-4:i=' + str(i - 4),

            '-4:len(syllable)=' + str(len(syllablen4)),

            '-4:syllable[:1]=' + syllablen4[:1],
            '-4:syllable[:2]=' + syllablen4[:2],
            '-4:syllable[-1:]=' + syllablen4[-1:],
            '-4:syllable[-2:]=' + syllablen4[-2:],
            '-4:syllable[-3:]=' + syllablen4[-3:],

            '-4:lh=' + lhn4,
            '-4:oc=' + ocn4,
            '-4:lpos=' + lposn4,
            '-4:wpos=' + wposn4,

        ])
    else:
        features.append('BOL')

    if i > 4:
        syllablen5 = line[i - 5][0]
        lhn5 = line[i - 5][1]
        ocn5 = line[i - 5] [2]
        lposn5 = line[i - 5][3]
        wposn5 = line[i - 5][4]
        features.extend([
            '-5:i=' + str(i - 5),

            '-5:len(syllable)=' + str(len(syllablen5)),

            '-5:syllable[:1]=' + syllablen5[:1],
            '-5:syllable[:2]=' + syllablen5[:2],
            '-5:syllable[-1:]=' + syllablen5[-1:],
            '-5:syllable[-2:]=' + syllablen5[-2:],
            '-5:syllable[-3:]=' + syllablen5[-3:],

            '-5:lh=' + lhn5,
            '-5:oc=' + ocn5,
            '-5:lpos=' + lposn5,
            '-5:wpos=' + wposn5,

        ])
    else:
        features.append('BOL')

    if i > 5:
        syllablen6 = line[i - 6][0]
        lhn6 = line[i - 6][1]
        ocn6 = line[i - 6][2]
        lposn6 = line[i - 6][3]
        wposn6 = line[i - 6][4]
        features.extend([
            '-6:i=' + str(i - 6),

            '-6:len(syllable)=' + str(len(syllablen6)),

            '-6:syllable[:1]=' + syllablen6[:1],
            '-6:syllable[:2]=' + syllablen6[:2],
            '-6:syllable[-1:]=' + syllablen6[-1:],
            '-6:syllable[-2:]=' + syllablen6[-2:],
            '-6:syllable[-3:]=' + syllablen6[-3:],

            '-6:lh=' + lhn6,
            '-6:oc=' + ocn6,
            '-6:lpos=' + lposn6,
            '-6:wpos=' + wposn6,

        ])
    else:
        features.append('BOL')

    if i > 6:
        syllablen7 = line[i - 7][0]
        lhn7 = line[i - 7][1]
        ocn7 = line[i - 7][2]
        lposn7 = line[i - 7][3]
        wposn7 = line[i - 7][4]
        features.extend([
            '-7:i=' + str(i - 7),

            '-7:len(syllable)=' + str(len(syllablen7)),

            '-7:syllable[:1]=' + syllablen7[:1],
            '-7:syllable[:2]=' + syllablen7[:2],
            '-7:syllable[-1:]=' + syllablen7[-1:],
            '-7:syllable[-2:]=' + syllablen7[-2:],
            '-7:syllable[-3:]=' + syllablen7[-3:],

            '-7:lh=' + lhn7,
            '-7:oc=' + ocn7,
            '-7:lpos=' + lposn7,
            '-7:wpos=' + wposn7,

        ])
    else:
        features.append('BOL')

    if i > 7:
        syllablen8 = line[i - 8][0]
        lhn8 = line[i - 8][1]
        ocn8 = line[i - 8][2]
        lposn8 = line[i - 8][3]
        wposn8 = line[i - 8][4]
        features.extend([
            '-8:i=' + str(i - 8),

            '-8:len(syllable)=' + str(len(syllablen8)),

            '-8:syllable[:1]=' + syllablen8[:1],
            '-8:syllable[:2]=' + syllablen8[:2],
            '-8:syllable[-1:]=' + syllablen8[-1:],
            '-8:syllable[-2:]=' + syllablen8[-2:],
            '-8:syllable[-3:]=' + syllablen8[-3:],

            '-8:lh=' + lhn8,
            '-8:oc=' + ocn8,
            '-8:lpos=' + lposn8,
            '-8:wpos=' + wposn8,

        ])
    else:
        features.append('BOL')

    if i > 8:
        syllablen9 = line[i - 9][0]
        lhn9 = line[i - 9][1]
        ocn9 = line[i - 9][2]
        lposn9 = line[i - 9][3]
        wposn9 = line[i - 9][4]
        features.extend([
            '-9:i=' + str(i - 9),

            '-9:len(syllable)=' + str(len(syllablen9)),

            '-9:syllable[:1]=' + syllablen9[:1],
            '-9:syllable[:2]=' + syllablen9[:2],
            '-9:syllable[-1:]=' + syllablen9[-1:],
            '-9:syllable[-2:]=' + syllablen9[-2:],
            '-9:syllable[-3:]=' + syllablen9[-3:],

            '-9:lh=' + lhn9,
            '-9:oc=' + ocn9,
            '-9:lpos=' + lposn9,
            '-9:wpos=' + wposn9,

        ])
    else:
        features.append('BOL')

    if i > 9:
        syllablen10 = line[i - 10][0]
        lhn10 = line[i - 10][1]
        ocn10 = line[i - 10][2]
        lposn10 = line[i - 10][3]
        wposn10 = line[i - 10][4]
        features.extend([
            '-10:i=' + str(i - 10),

            '-10:len(syllable)=' + str(len(syllablen10)),

            '-10:syllable[:1]=' + syllablen10[:1],
            '-10:syllable[:2]=' + syllablen10[:2],
            '-10:syllable[-1:]=' + syllablen10[-1:],
            '-10:syllable[-2:]=' + syllablen10[-2:],
            '-10:syllable[-3:]=' + syllablen10[-3:],

            '-10:lh=' + lhn10,
            '-10:oc=' + ocn10,
            '-10:lpos=' + lposn10,
            '-10:wpos=' + wposn10,

        ])
    else:
        features.append('BOL')

    if i < len(line) - 1:  # following syllable
        syllable1 = line[i + 1][0]
        lh1 = line[i + 1][1]
        oc1 = line[i + 1][2]
        lpos1 = line[i + 1][3]
        wpos1 = line[i + 1][4]
        features.extend([
            '+1:i=' + str(i + 1),

            '+1:len(syllable)=' + str(len(syllable1)),

            '+1:syllable[:1]=' + syllable1[:1],
            '+1:syllable[:2]=' + syllable1[:2],
            '+1:syllable[-1:]=' + syllable1[-1:],
            '+1:syllable[-2:]=' + syllable1[-2:],
            '+1:syllable[-3:]=' + syllable1[-3:],

            '(-1:syllable[-1] + syllable[0])=%s' +
            (syllable[-2:] + syllable1[:2]),

            '+1:lh=' + lh1,
            '+1:oc=' + oc1,
            '+1:lpos=' + lpos1,
            '+1:wpos=' + wpos1,

        ])
    else:
        features.append('EOL')

    if i < len(line) - 2:
        syllable2 = line[i + 2][0]
        lh2 = line[i + 2][1]
        oc2 = line[i + 2][2]
        lpos2 = line[i + 2][3]
        wpos2 = line[i + 2][4]
        features.extend([
            '+2:i=' + str(i + 2),

            '+2:len(syllable)=' + str(len(syllable2)),

            '+2:syllable[:1]=' + syllable2[:1],
            '+2:syllable[:2]=' + syllable2[:2],
            '+2:syllable[-1:]=' + syllable2[-1:],
            '+2:syllable[-2:]=' + syllable2[-2:],
            '+2:syllable[-3:]=' + syllable2[-3:],

            '+2:lh=' + lh2,
            '+2:oc=' + oc2,
            '+2:lpos=' + lpos2,
            '+2:wpos=' + wpos2,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 3:
        syllable3 = line[i + 3][0]
        lh3 = line[i + 3][1]
        oc3 = line[i + 3][2]
        lpos3 = line[i + 3][3]
        wpos3 = line[i + 3][4]
        features.extend([
            '+3:i=' + str(i + 3),

            '+3:len(syllable)=' + str(len(syllable3)),

            '+3:syllable[:1]=' + syllable3[:1],
            '+3:syllable[:2]=' + syllable3[:2],
            '+3:syllable[-1:]=' + syllable3[-1:],
            '+3:syllable[-2:]=' + syllable3[-2:],
            '+3:syllable[-3:]=' + syllable3[-3:],

            '+3:lh=' + lh3,
            '+3:oc=' + oc3,
            '+3:lpos=' + lpos3,
            '+3:wpos=' + wpos3,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 4:
        syllable4 = line[i + 4][0]
        lh4 = line[i + 4][1]
        oc4 = line[i + 4][2]
        lpos4 = line[i + 4][3]
        wpos4 = line[i + 4][4]
        features.extend([
            '+4:i=' + str(i + 4),

            '+4:len(syllable)=' + str(len(syllable4)),

            '+4:syllable[:1]=' + syllable4[:1],
            '+4:syllable[:2]=' + syllable4[:2],
            '+4:syllable[-1:]=' + syllable4[-1:],
            '+4:syllable[-2:]=' + syllable4[-2:],
            '+4:syllable[-3:]=' + syllable4[-3:],

            '+4:lh=' + lh4,
            '+4:oc=' + oc4,
            '+4:lpos=' + lpos4,
            '+4:wpos=' + wpos4,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 5:
        syllable5 = line[i + 5][0]
        lh5 = line[i + 5][1]
        oc5 = line[i + 5][2]
        lpos5 = line[i + 5][3]
        wpos5 = line[i + 5][4]
        features.extend([
            '+5:i=' + str(i + 5),

            '+5:len(syllable)=' + str(len(syllable5)),

            '+5:syllable[:1]=' + syllable5[:1],
            '+5:syllable[:2]=' + syllable5[:2],
            '+5:syllable[-1:]=' + syllable5[-1:],
            '+5:syllable[-2:]=' + syllable5[-2:],
            '+5:syllable[-3:]=' + syllable5[-3:],

            '+5:lh=' + lh5,
            '+5:oc=' + oc5,
            '+5:lpos=' + lpos5,
            '+5:wpos=' + wpos5,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 6:
        syllable6 = line[i + 6][0]
        lh6 = line[i + 6][1]
        oc6 = line[i + 6][2]
        lpos6 = line[i + 6][3]
        wpos6 = line[i + 6][4]
        features.extend([
            '+6:i=' + str(i + 6),

            '+6:len(syllable)=' + str(len(syllable6)),

            '+6:syllable[:1]=' + syllable6[:1],
            '+6:syllable[:2]=' + syllable6[:2],
            '+6:syllable[-1:]=' + syllable6[-1:],
            '+6:syllable[-2:]=' + syllable6[-2:],
            '+6:syllable[-3:]=' + syllable6[-3:],

            '+6:lh=' + lh6,
            '+6:oc=' + oc6,
            '+6:lpos=' + lpos6,
            '+6:wpos=' + wpos6,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 7:
        syllable7 = line[i + 7][0]
        lh7 = line[i + 7][1]
        oc7 = line[i + 7][2]
        lpos7 = line[i + 7][3]
        wpos7 = line[i + 7][4]
        features.extend([
            '+7:i=' + str(i + 7),

            '+7:len(syllable)=' + str(len(syllable7)),

            '+7:syllable[:1]=' + syllable7[:1],
            '+7:syllable[:2]=' + syllable7[:2],
            '+7:syllable[-1:]=' + syllable7[-1:],
            '+7:syllable[-2:]=' + syllable7[-2:],
            '+7:syllable[-3:]=' + syllable7[-3:],

            '+7:lh=' + lh7,
            '+7:oc=' + oc7,
            '+7:lpos=' + lpos7,
            '+7:wpos=' + wpos7,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 8:
        syllable8 = line[i + 8][0]
        lh8 = line[i + 8][1]
        oc8 = line[i + 8][2]
        lpos8 = line[i + 8][3]
        wpos8 = line[i + 8][4]
        features.extend([
            '+8:i=' + str(i + 8),

            '+8:len(syllable)=' + str(len(syllable8)),

            '+8:syllable[:1]=' + syllable8[:1],
            '+8:syllable[:2]=' + syllable8[:2],
            '+8:syllable[-1:]=' + syllable8[-1:],
            '+8:syllable[-2:]=' + syllable8[-2:],
            '+8:syllable[-3:]=' + syllable8[-3:],

            '+8:lh=' + lh8,
            '+8:oc=' + oc8,
            '+8:lpos=' + lpos8,
            '+8:wpos=' + wpos8,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 9:
        syllable9 = line[i + 9][0]
        lh9 = line[i + 9][1]
        oc9 = line[i + 9][2]
        lpos9 = line[i + 9][3]
        wpos9 = line[i + 9][4]
        features.extend([
            '+9:i=' + str(i + 9),

            '+9:len(syllable)=' + str(len(syllable9)),

            '+9:syllable[:1]=' + syllable9[:1],
            '+9:syllable[:2]=' + syllable9[:2],
            '+9:syllable[-1:]=' + syllable9[-1:],
            '+9:syllable[-2:]=' + syllable9[-2:],
            '+9:syllable[-3:]=' + syllable9[-3:],

            '+9:lh=' + lh9,
            '+9:oc=' + oc9,
            '+9:lpos=' + lpos9,
            '+9:wpos=' + wpos9,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 10:
        syllable10 = line[i + 10][0]
        lh10 = line[i + 10][1]
        oc10 = line[i + 10][2]
        lpos10 = line[i + 10][3]
        wpos10 = line[i + 10][4]
        features.extend([
            '+10:i=' + str(i + 10),

            '+10:len(syllable)=' + str(len(syllable10)),

            '+10:syllable[:1]=' + syllable10[:1],
            '+10:syllable[:2]=' + syllable10[:2],
            '+10:syllable[-1:]=' + syllable10[-1:],
            '+10:syllable[-2:]=' + syllable10[-2:],
            '+10:syllable[-3:]=' + syllable10[-3:],

            '+10:lh=' + lh10,
            '+10:oc=' + oc10,
            '+10:lpos=' + lpos10,
            '+10:wpos=' + wpos10,
        ])
    else:
        features.append('EOL')

    return features

##########################################################################


def line2features(line):
    return [syllable2features(line, i) for i in range(len(line))]


def line2labels(line):
    return [label for token, lh, oc, lpos1, wpos1, label in line]


def line2tokens(line):
    return [token for token, lh, oc, lpos1, wpos1, label in line]


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
    trainer.train('MHGMETRICS_dev.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('MHGMETRICS_dev.crfsuite')

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

    y_pred = [tagger.tag(xseq) for xseq in X_test]

    bioc = bio_classification_report(y_test, y_pred)

    # to parse
    p, r, f1, s = bioc[0]
    print(p)
    print(r)
    print(f1)
    print(s)
    tot_avgs = []

    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        tot_avgs.append(v)
    print(tot_avgs)
    # toext = [0] * (len(s) - 3)
    # tot_avgs.extend(toext)
    # print(tot_avgs)
    tot_avgs = tot_avgs[:2]
    all_s = [sum(s)] * len(s)
    print(all_s)

    rep = bioc[1]
    all_labels = []
    for word in rep.split():
        if word.isupper():
            all_labels.append(word)

    # print(bio_classification_report(y_test, y_pred)[1])

    data = {"labels": all_labels, "precision": p, "recall": r,
            "f1": f1, "support": s, "tots": tot_avgs, "all_s": all_s}

    df = pd.DataFrame(data)

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
