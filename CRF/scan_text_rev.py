# -*- coding: utf-8 -*-
from __future__ import unicode_literals  # for python2 compatibility
from __future__ import division
from __future__ import absolute_import

# created at UC Berkeley 2015
# Authors: Christopher Hench

# This program scans MHG epic poetry, returning data to analyze statistically

import codecs
import pycrfsuite
import numpy as np
import itertools


import codecs
import pycrfsuite
import numpy as np
from get_features import get_features
import itertools
from get_features import syllableweight


def only_four_stresses(lines_w_features, tagger, sylls):
    labs = ["MORA_HAUPT", "MORA", "DOPPEL", "HALB_HAUPT", "HALB", "EL"]
    stressed = ["MORA_HAUPT", "DOPPEL", "HALB_HAUPT"]
    four_stress = []

    for i, line in enumerate(lines_w_features):

        t_line = tagger.tag(line)

        line_sylls = sylls[i]

        stress = 0
        for i2, l in enumerate(t_line):

                # increment stress
            if t_line[i2] in stressed:
                stress += 1

            # no doppel can be light
            if l == "DOPPEL" and syllableweight(line_sylls[i2]) == "L":
                stress += 5  # auto sends to reweight probs

            # no halb-haupt can be heavy
            if l == "HALB_HAUPT" and syllableweight(line_sylls[i2]) == "H":
                stress += 5  # auto sends to reweight probs

            # no EL can be heavy
            if l == "EL" and syllableweight(line_sylls[i2]) == "H":
                stress += 5  # auto sends to reweight probs

            # error alternation, recount
            accs = ["MORA_HAUPT", "HALB_HAUPT"]
            if i2 > 0 and l in accs:  # rule out if two
                if t_line[i2 - 1] in accs:
                    stress += 5

            # rule out if no stress following doppel
            if i2 < len(t_line) - 1 and l == "DOPPEL":
                if t_line[i2 + 1] not in stressed:
                    stress += 5

            if i2 > 0 and l == "DOPPEL":  # rule out stress before double
                if t_line[i2 - 1] in accs:
                    stress += 5

            if 0 < i2 < len(t_line) - 1 and l == "EL":
                if t_line[i2 - 1] in accs and t_line[i2 + 1] in accs:
                    stress += 5

        # if > 4 stresses, look at probs
        if stress != 4:
            line_probs = []
            for i3, l in enumerate(t_line):

                # marginal probablities
                probs = [(lb, tagger.marginal(lb, i3)) for lb in labs]

                # no doppel can be light
                if syllableweight(line_sylls[i3]) == "L":
                    probs = [x for x in probs if x[0] != "DOPPEL"]

                # no halbhaupt or EL can be heavy
                if syllableweight(line_sylls[i3]) == "H":
                    probs = [x for x in probs if x[0] != "HALB_HAUPT"]
                    probs = [x for x in probs if x[0] != "EL"]

                probs = sorted(probs, key=lambda tup: tup[1], reverse=True)

                # if very certain, only take top 2, otherwise top 4
                if probs[0][1] > .9:
                    line_probs.append(probs[:2])
                else:
                    line_probs.append(probs[:4])

            # get combinations of syll values with probs
            combos = itertools.product(*line_probs)

            # verify each combo and rank
            final_line = (0, 0)
            for c in combos:
                stress = 0
                tot_prob = 0
                for i4, l in enumerate(c):
                    tot_prob += l[1]
                    if l[0] in stressed:
                        stress += 1
                    if i4 < (
                            len(c) - 1) and l[0] == "DOPPEL":  # rule out if no stress after double
                        if c[i4 + 1][0] not in stressed:
                            stress += 5

                    # error alternation recount
                    accs = ["MORA_HAUPT", "HALB_HAUPT"]
                    if i4 > 0 and l[0] in accs:  # rule out if two
                        if c[i4 - 1][0] in accs:
                            stress += 5

                    if i4 > 0 and l[0] == "DOPPEL":  # no stress b4 double
                        if c[i4 - 1][0] in accs:
                            stress += 5

                    if 0 < i4 < len(c) - 1 and l == "EL":
                        if c[i4 - 1] in accs and c[i4 + 1] in accs:
                            stress += 5

                if stress == 4 and tot_prob > final_line[1]:
                    final_line = (c, tot_prob)

            try:
                t_line = [x[0] for x in final_line[0]]
            except TypeError:
                pass  # not continue, pass will do nothing

        four_stress.append(t_line)  # will take orig if no errors

    # additional fixes
    final_labels = []
    for line in four_stress:
        count = 0
        new_line = []
        for i, l in enumerate(line):

            # fix /  X' ◡   /
            if (0 < i < (len(line) - 1) and
                    line[i - 1] == "MORA_HAUPT" and
                    l == "HALB" and
                    line[i + 1] in stressed):
                new_line.append("MORA")
            else:
                new_line.append(l)
        final_labels.append(new_line)

    final_labels2 = []
    for line in final_labels:
        new_line = []
        for i, l in enumerate(line):

            # fix /  X' ◡ X  / X'
            if (1 < i < (len(line) - 1) and
                    line[i - 1] == "HALB" and
                    line[i - 2] == "MORA_HAUPT" and
                    l == "MORA" and
                    line[i + 1] in stressed):
                new_line.append("HALB")
            else:
                new_line.append(l)
        final_labels2.append(new_line)

    final_labels = final_labels2

    final_labels2 = []
    for line in final_labels:
        new_line = []
        for i, l in enumerate(line):

            # fix /  X' X ◡  / X'
            if (0 < i < (len(line) - 2) and
                    line[i - 1] == "MORA_HAUPT" and
                    l == "MORA" and
                    line[i + 1] == "HALB" and
                    line[i + 2] in stressed):
                new_line.append("HALB")
            else:
                new_line.append(l)
        final_labels2.append(new_line)

    final_labels = final_labels2

    final_labels2 = []
    for line in final_labels:
        new_line = line
        for i, l in enumerate(line):

            # fix /  X' X X  / X'
            if (0 < i < (len(line) - 2) and
                    line[i - 1] == "MORA_HAUPT" and
                    l == "MORA" and
                    line[i + 1] == "MORA" and
                    line[i + 2] in stressed):

                new_line[i] = "HALB"
                new_line[i + 1] = "HALB"

        final_labels2.append(new_line)

    final_labels = final_labels2

    # # zweisilbig maennlich kadenz
    # final_labels2 = []
    # prefixes = ["ge", "be", "en"]
    # for i, line in enumerate(final_labels):
    #     new_line = line
    #     if line[-4:] == ["MORA_HAUPT", "HALB", "HALB", "MORA_HAUPT"]:
    #         if len(sylls[i][-1]) > 1 and sylls[
    #                 i][-1][-2] not in prefixes and syllableweight(sylls[i][-1][-2]) == "L":
    #             new_line[-4:] = ["MORA_HAUPT", "MORA", "HALB_HAUPT", "HALB"]

    #     final_labels2.append(new_line)

    # final_labels = final_labels2

    return(final_labels)
