# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

# Created at UC Berkeley 2015
# Authors: Christopher Hench
# ==============================================================================
'''This code presents summary statistics for MHG syllables used for scansion
based on the paper presented at the NAACL-CLFL 2016 by Christopher Hench and
Alex Estes.'''

import pandas as pd
from nltk.tag.util import str2tuple

with open("Data/CLFL_all_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

lines = data.split('\n')
tags = [[str2tuple(x) for x in line.split()] for line in lines]
tags = [[x[0] for x in line] for line in tags]

all_lines = []
all_sylls = []
for line in tags:
    newline = []
    word = ""
    l_syllables = 0
    s_line = []
    s_word = []
    for syll in line:
        if syll == "WBY":
            newline.append(word)
            s_line.append(s_word)
            word = ""
            s_word = []
        else:
            l_syllables += 1
            word += syll
            s_word.append(syll)

    newline.append(word)
    s_line.append(s_word)
    newline = [x for x in newline if len(x) > 0]
    all_lines.append((newline, l_syllables))
    all_sylls.append(s_line)

all_lines = [x for x in all_lines if x[1] != 0]
all_sylls = [[x for x in line if len(x) != 0]
             for line in all_sylls if len(line) != 0]

w_len_lines = [len(x) for x in all_sylls]
sylls_words = [item for sublist in all_sylls for item in sublist]
l_sylls_words = [len(x) for x in sylls_words]
c_sylls_words = [len(''.join(x)) for x in sylls_words]

line_lengths = [x[1] for x in all_lines]
line_words = [x[0] for x in all_lines]

joined_lines = [''.join(line) for line in line_words]
jl_lengths = [len(x) for x in joined_lines]

data = {
    "syll_line": line_lengths,
    "char_line": jl_lengths,
    "word_line": w_len_lines}

# summary statistics
df = pd.DataFrame(data)
df.char_line.sum(axis=0)
sum(jl_lengths)

data2 = {"sylls/words": l_sylls_words, "char/words": c_sylls_words}
df2 = pd.DataFrame(data2)

only_sylls = [item for sublist in sylls_words for item in sublist]
l_only_sylls = [len(x) for x in only_sylls]
data3 = {"char/sylls": l_only_sylls}
df3 = pd.DataFrame(data3)

print(df.describe())
print(df2.describe())
print(df3.describe())
