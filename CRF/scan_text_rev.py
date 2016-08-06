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


def only_four_stresses(tags, tagger):
    labs = ["MORA_HAUPT", "MORA", "DOPPEL", "HALB_HAUPT", "HALB", "EL"]
    stressed = ["MORA_HAUPT", "DOPPEL", "HALB_HAUPT"]
    four_stress = []
    for line in tags:
        t_line = tagger.tag(line)
        stress = 0
        for l in t_line:
            if l in stressed:
                stress += 1
        if stress > 4:
            l_n_s = []
            # take four likeliest stresses, everyhting else make second prob
            top_probs = []
            for i, l in enumerate(t_line):
                if l in stressed:
                    top_probs.append((i, tagger.marginal(l, i)))

            top_probs = sorted(top_probs, key=lambda tup: tup[1], reverse=True)
            to_change = [i for i in top_probs[4:]]

            for i, l in enumerate(t_line):
                new_l = l
                if i in [x[0] for x in to_change]:
                    prob_tups = []
                    for lb in labs:
                        prob_tups.append((lb, tagger.marginal(lb, i)))
                    sorted_probs = sorted(prob_tups, key=lambda tup: tup[1],
                                          reverse=True)
                    sorted_probs = [x for x in sorted_probs if x[0] not in
                                    stressed]
                    if len(sorted_probs) > 0:
                        new_l = sorted_probs[0][0]
                    else:
                        new_l = l

                if tagger.marginal(l, i) < .95 and l not in stressed:
                    prob_tups = []
                    for lb in labs:
                        prob_tups.append((lb, tagger.marginal(lb, i)))
                    sorted_probs = sorted(prob_tups, key=lambda tup: tup[1],
                                          reverse=True)
                    sorted_probs = [x for x in sorted_probs if x[0] not in
                                    stressed]
                    new_l = sorted_probs[1][0]

                l_n_s.append(new_l)

            four_stress.append(l_n_s)

        else:
            four_stress.append(t_line)

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

    return(final_labels)
