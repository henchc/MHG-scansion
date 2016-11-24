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


def only_four_stresses(lines_w_features, tagger):
    labs = ["MORA_HAUPT", "MORA", "DOPPEL", "HALB_HAUPT", "HALB", "EL"]
    stressed = ["MORA_HAUPT", "DOPPEL", "HALB_HAUPT"]
    four_stress = []
    for line in lines_w_features:

        t_line = tagger.tag(line)

        stress = 0
        for l in t_line:
            if l in stressed:
                stress += 1

        if stress != 4:
            line_probs = []
            for i, l in enumerate(t_line):
                probs = [(l, tagger.marginal(l, i)) for l in labs]
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
                for l in c:
                    tot_prob += l[1]
                    if l[0] in stressed:
                        stress += 1

                if stress == 4 and tot_prob > final_line[1]:
                    final_line = (c, tot_prob)

            try:
                t_line = [x[0] for x in final_line[0]]
            except TypeError:
                continue

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

    return(final_labels)
