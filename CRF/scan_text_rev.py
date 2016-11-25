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

            # no doppel can be light
            if l == "DOPPEL" and syllableweight(line_sylls[i2]) == "L":
                # unaccented so it get sent to recalculate without doppel
                stress += 5

            if t_line[i2] in stressed:
                stress += 1

            if i2 > 0 and l == "MORA_HAUPT":  # rule out if two
                if t_line[i2 - 1] == "MORA_HAUPT":
                    stress += 5

            if i2 < len(t_line) - 1 and l == "DOPPEL":
                if t_line[i2 + 1] not in stressed:
                    stress += 5

        if stress != 4:
            line_probs = []
            for i3, l in enumerate(t_line):
                probs = [(l, tagger.marginal(l, i3)) for l in labs]

                # no doppel can be light
                if syllableweight(line_sylls[i3]) == "L":
                    probs = [x for x in probs if x[0] != "DOPPEL"]

                probs = sorted(probs, key=lambda tup: tup[1], reverse=True)
                if probs[0][1] > .9:
                    line_probs.append(probs[:2])
                else:
                    line_probs.append(probs[:4])

            combos = itertools.product(*line_probs)

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

                    if i4 > 0 and l[0] == "MORA_HAUPT":  # rule out if two
                        if c[i4 - 1][0] == "MORA_HAUPT":
                            stress += 5

                if stress == 4 and tot_prob > final_line[1]:
                    final_line = (c, tot_prob)

            try:
                t_line = [x[0] for x in final_line[0]]
            except TypeError:
                pass

        four_stress.append(t_line)

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

    # # fix endings with light halves
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
