# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import pickle


def mhgscansion(taggedtuples):

    alllines = []
    count2 = 1

    for line in taggedtuples:
        scan = ""
        text = ""
        cadence = ""
        stresscount = 0
        text += str(count2) + ": "
        scan += ((len(str(count2)) + 2) * ' ')
        for word in line:
            count = 0
            for tup in word:
                if tup[1] == "MORA_HAUPT":
                    mark = "  /  X' "
                    stresscount += 1
                elif tup[1] == "MORA_NEBEN":
                    mark = "  /  X` "
                    stresscount += 1
                elif tup[1] == "MORA":
                    mark = "X "
                elif tup[1] == "DOPPEL":
                    mark = "  /  ---'"
                    stresscount += 1
                elif tup[1] == "EL":
                    mark = "* "
                elif tup[1] == "HALB":
                    mark = "◡ "
                elif tup[1] == "HALB_HAUPT":
                    mark = "  /  ◡'' "
                    stresscount += 1
                elif tup[1] == "HALB_NEBEN":
                    mark = "  /  ◡` "
                    stresscount += 1
                else:
                    mark = ""

                scan += mark

                if tup[1] == "EL":
                    text += tup[0][:-1]

                elif count < len(word) - 1:
                    text += tup[0] + "-"

                else:
                    text += tup[0]

                count += 1
            text += " "

        if stresscount == 3:
            cadence = "Stumpf"

        # else:
        #     if len(line) > 3:
        #         if line[-3][1] == "MORA_HAUPT":
        #             cadence = "Einsilbig männlich"
        #         elif line[-3][1] == "HALB":
        #             cadence = "Zweisilbig männlich"
        #         elif line[-3][1] == "MORA":
        #             cadence = "Zweisilbig weiblich"
        #         elif line[-3][1] == "MORA_NEBEN" and line[-5][1] == "WBY":
        #             cadence = "Zweisilbig klingend"
        #         # elif line[-3][1] == "MORA_NEBEN" and line[-6][1] == "WBY":
        #         # 	cadence = "Dreisilbig klingend"
            #
        #         else:
        #             cadence = "Dreisilbig klingend"

        scan += "   "

        alllines.append((text, scan))
        count2 += 1

    to_write = ""
    for line in alllines:
        to_write += line[0] + '\n' + line[1] + '\n'

    return(to_write)
