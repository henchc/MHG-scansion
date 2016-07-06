# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import nltk
import nltk.tag


# with open("handtagged.txt", 'r', encoding='utf-8') as f:
# 	tagged = f.read() #text must be clean

# tagged = tagged.split('\n')


# newtagged = []
# for line in tagged:
# 	news = [nltk.tag.str2tuple(t) for t in line.split()]
# 	if news != []:
# 		newtagged.append(news)


def mhgscansion (taggedtuples):

	alllines = []
	count2 = 1

	for line in taggedtuples:
		scan = ""
		text = ""
		cadence = ""
		count = 0
		stresscount = 0
		text += str(count2) + ": "
		for tup in line:
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
				stresscount +=1
			elif tup[1] == "HALB_NEBEN":
				mark = "  /  ◡` "
				stresscount +=1
			else:
				mark = ""

			scan += mark

			if tup[0] != "WBY" and tup[0] != "BEGL" and tup[0] != "ENDL":
				if tup[1] == "EL":
					text += tup[0][:-1]

				elif count < len(line)-1 and line[count+1][0] != "WBY":
					text += tup[0] + "-"

				else:
					text += tup[0] + " "

			count +=1



		if stresscount == 3:
			cadence = "Stumpf"

		else:
			if len(line) > 3:
				if line[-3][1] == "MORA_HAUPT":
					cadence = "Einsilbig männlich"
				elif line[-3][1] == "HALB":
					cadence = "Zweisilbig männlich"
				elif line[-3][1] == "MORA":
					cadence = "Zweisilbig weiblich"
				elif line[-3][1] == "MORA_NEBEN" and line[-5][1] == "WBY":
					cadence = "Zweisilbig klingend"
				# elif line[-3][1] == "MORA_NEBEN" and line[-6][1] == "WBY":
				# 	cadence = "Dreisilbig klingend" 

				else:
					cadence = "Dreisilbig klingend"

		scan += "   " + cadence

		alllines.append((text,scan))
		count2 += 1

	return(alllines)


# testlines = mhgscansion(newtagged)
# for line in testlines:
# 	print (line[0] + '\n' + line[1] + '\n')

