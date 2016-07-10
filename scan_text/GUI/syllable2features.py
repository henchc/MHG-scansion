
def syllable2features(line, i):
    syllable = line[i][0]
    lhoc = line[i][1]  # light, heavy, open, closed
    lpos = line[i][2]  # position in line
    wpos = line[i][3]  # position in word

    features = [
        'bias',  # bias to current syllable
        'i=' + str(i),  # position in line

        'len(syllable)=' + str(len(syllable)),  # length of syllable

        'syllable[:1]=' + syllable[:1],  # first letter
        'syllable[:2]=' + syllable[:2],  # first two letters
        'syllable[-1:]=' + syllable[-1:],  # last letter
        'syllable[-2:]=' + syllable[-2:],  # last two letters
        'syllable[-3:]=' + syllable[-3:],  # last threel letters


        'lhoc=' + lhoc,
        'lpos=' + lpos,
        'wpos=' + wpos,

    ]
    if i > 0:  # preceding syllable
        syllablen1 = line[i - 1][0]
        lhocn1 = line[i - 1][1]
        lposn1 = line[i - 1][2]
        wposn1 = line[i - 1][3]
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

            '-1:lhoc=' + lhocn1,
            '-1:lpos=' + lposn1,
            '-1:wpos=' + wposn1,
        ])
    else:
        features.append('BOL')

    if i > 1:
        syllablen2 = line[i - 2][0]
        lhocn2 = line[i - 2][1]
        lposn2 = line[i - 2][2]
        wposn2 = line[i - 2][3]
        features.extend([
            '-2:i=' + str(i - 2),

            '-2:len(syllable)=' + str(len(syllablen2)),

            '-2:syllable[:1]=' + syllablen2[:1],
            '-2:syllable[:2]=' + syllablen2[:2],
            '-2:syllable[-1:]=' + syllablen2[-1:],
            '-2:syllable[-2:]=' + syllablen2[-2:],
            '-2:syllable[-3:]=' + syllablen2[-3:],

            '-2:lhoc=' + lhocn2,
            '-2:lpos=' + lposn2,
            '-2:wpos=' + wposn2,

        ])
    else:
        features.append('BOL')

    if i > 2:
        syllablen3 = line[i - 3][0]
        lhocn3 = line[i - 3][1]
        lposn3 = line[i - 3][2]
        wposn3 = line[i - 3][3]
        features.extend([
            '-3:i=' + str(i - 3),

            '-3:len(syllable)=' + str(len(syllablen3)),

            '-3:syllable[:1]=' + syllablen3[:1],
            '-3:syllable[:2]=' + syllablen3[:2],
            '-3:syllable[-1:]=' + syllablen3[-1:],
            '-3:syllable[-2:]=' + syllablen3[-2:],
            '-3:syllable[-3:]=' + syllablen3[-3:],

            '-3:lhoc=' + lhocn3,
            '-3:lpos=' + lposn3,
            '-3:wpos=' + wposn3,

        ])
    else:
        features.append('BOL')

    if i > 3:
        syllablen4 = line[i - 4][0]
        lhocn4 = line[i - 4][1]
        lposn4 = line[i - 4][2]
        wposn4 = line[i - 4][3]
        features.extend([
            '-4:i=' + str(i - 4),

            '-4:len(syllable)=' + str(len(syllablen4)),

            '-4:syllable[:1]=' + syllablen4[:1],
            '-4:syllable[:2]=' + syllablen4[:2],
            '-4:syllable[-1:]=' + syllablen4[-1:],
            '-4:syllable[-2:]=' + syllablen4[-2:],
            '-4:syllable[-3:]=' + syllablen4[-3:],

            '-4:lhoc=' + lhocn4,
            '-4:lpos=' + lposn4,
            '-4:wpos=' + wposn4,

        ])
    else:
        features.append('BOL')

    if i > 4:
        syllablen5 = line[i - 5][0]
        lhocn5 = line[i - 5][1]
        lposn5 = line[i - 5][2]
        wposn5 = line[i - 5][3]
        features.extend([
            '-5:i=' + str(i - 5),

            '-5:len(syllable)=' + str(len(syllablen5)),

            '-5:syllable[:1]=' + syllablen5[:1],
            '-5:syllable[:2]=' + syllablen5[:2],
            '-5:syllable[-1:]=' + syllablen5[-1:],
            '-5:syllable[-2:]=' + syllablen5[-2:],
            '-5:syllable[-3:]=' + syllablen5[-3:],

            '-5:lhoc=' + lhocn5,
            '-5:lpos=' + lposn5,
            '-5:wpos=' + wposn5,

        ])
    else:
        features.append('BOL')

    if i > 5:
        syllablen6 = line[i - 6][0]
        lhocn6 = line[i - 6][1]
        lposn6 = line[i - 6][2]
        wposn6 = line[i - 6][3]
        features.extend([
            '-6:i=' + str(i - 6),

            '-6:len(syllable)=' + str(len(syllablen6)),

            '-6:syllable[:1]=' + syllablen6[:1],
            '-6:syllable[:2]=' + syllablen6[:2],
            '-6:syllable[-1:]=' + syllablen6[-1:],
            '-6:syllable[-2:]=' + syllablen6[-2:],
            '-6:syllable[-3:]=' + syllablen6[-3:],

            '-6:lhoc=' + lhocn6,
            '-6:lpos=' + lposn6,
            '-6:wpos=' + wposn6,

        ])
    else:
        features.append('BOL')

    if i > 6:
        syllablen7 = line[i - 7][0]
        lhocn7 = line[i - 7][1]
        lposn7 = line[i - 7][2]
        wposn7 = line[i - 7][3]
        features.extend([
            '-7:i=' + str(i - 7),

            '-7:len(syllable)=' + str(len(syllablen7)),

            '-7:syllable[:1]=' + syllablen7[:1],
            '-7:syllable[:2]=' + syllablen7[:2],
            '-7:syllable[-1:]=' + syllablen7[-1:],
            '-7:syllable[-2:]=' + syllablen7[-2:],
            '-7:syllable[-3:]=' + syllablen7[-3:],

            '-7:lhoc=' + lhocn7,
            '-7:lpos=' + lposn7,
            '-7:wpos=' + wposn7,

        ])
    else:
        features.append('BOL')

    if i > 7:
        syllablen8 = line[i - 8][0]
        lhocn8 = line[i - 8][1]
        lposn8 = line[i - 8][2]
        wposn8 = line[i - 8][3]
        features.extend([
            '-8:i=' + str(i - 8),

            '-8:len(syllable)=' + str(len(syllablen8)),

            '-8:syllable[:1]=' + syllablen8[:1],
            '-8:syllable[:2]=' + syllablen8[:2],
            '-8:syllable[-1:]=' + syllablen8[-1:],
            '-8:syllable[-2:]=' + syllablen8[-2:],
            '-8:syllable[-3:]=' + syllablen8[-3:],

            '-8:lhoc=' + lhocn8,
            '-8:lpos=' + lposn8,
            '-8:wpos=' + wposn8,

        ])
    else:
        features.append('BOL')

    if i > 8:
        syllablen9 = line[i - 9][0]
        lhocn9 = line[i - 9][1]
        lposn9 = line[i - 9][2]
        wposn9 = line[i - 9][3]
        features.extend([
            '-9:i=' + str(i - 9),

            '-9:len(syllable)=' + str(len(syllablen9)),

            '-9:syllable[:1]=' + syllablen9[:1],
            '-9:syllable[:2]=' + syllablen9[:2],
            '-9:syllable[-1:]=' + syllablen9[-1:],
            '-9:syllable[-2:]=' + syllablen9[-2:],
            '-9:syllable[-3:]=' + syllablen9[-3:],

            '-9:lhoc=' + lhocn9,
            '-9:lpos=' + lposn9,
            '-9:wpos=' + wposn9,

        ])
    else:
        features.append('BOL')

    if i > 9:
        syllablen10 = line[i - 10][0]
        lhocn10 = line[i - 10][1]
        lposn10 = line[i - 10][2]
        wposn10 = line[i - 10][3]
        features.extend([
            '-10:i=' + str(i - 10),

            '-10:len(syllable)=' + str(len(syllablen10)),

            '-10:syllable[:1]=' + syllablen10[:1],
            '-10:syllable[:2]=' + syllablen10[:2],
            '-10:syllable[-1:]=' + syllablen10[-1:],
            '-10:syllable[-2:]=' + syllablen10[-2:],
            '-10:syllable[-3:]=' + syllablen10[-3:],

            '-10:lhoc=' + lhocn10,
            '-10:lpos=' + lposn10,
            '-10:wpos=' + wposn10,

        ])
    else:
        features.append('BOL')

    if i < len(line) - 1:  # following syllable
        syllable1 = line[i + 1][0]
        lhoc1 = line[i + 1][1]
        lpos1 = line[i + 1][2]
        wpos1 = line[i + 1][3]
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

            '+1:lhoc=' + lhoc1,
            '+1:lpos=' + lpos1,
            '+1:wpos=' + wpos1,

        ])
    else:
        features.append('EOL')

    if i < len(line) - 2:
        syllable2 = line[i + 2][0]
        lhoc2 = line[i + 2][1]
        lpos2 = line[i + 2][2]
        wpos2 = line[i + 2][3]
        features.extend([
            '+2:i=' + str(i + 2),

            '+2:len(syllable)=' + str(len(syllable2)),

            '+2:syllable[:1]=' + syllable2[:1],
            '+2:syllable[:2]=' + syllable2[:2],
            '+2:syllable[-1:]=' + syllable2[-1:],
            '+2:syllable[-2:]=' + syllable2[-2:],
            '+2:syllable[-3:]=' + syllable2[-3:],

            '+2:lhoc=' + lhoc2,
            '+2:lpos=' + lpos2,
            '+2:wpos=' + wpos2,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 3:
        syllable3 = line[i + 3][0]
        lhoc3 = line[i + 3][1]
        lpos3 = line[i + 3][2]
        wpos3 = line[i + 3][3]
        features.extend([
            '+3:i=' + str(i + 3),

            '+3:len(syllable)=' + str(len(syllable3)),

            '+3:syllable[:1]=' + syllable3[:1],
            '+3:syllable[:2]=' + syllable3[:2],
            '+3:syllable[-1:]=' + syllable3[-1:],
            '+3:syllable[-2:]=' + syllable3[-2:],
            '+3:syllable[-3:]=' + syllable3[-3:],

            '+3:lhoc=' + lhoc3,
            '+3:lpos=' + lpos3,
            '+3:wpos=' + wpos3,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 4:
        syllable4 = line[i + 4][0]
        lhoc4 = line[i + 4][1]
        lpos4 = line[i + 4][2]
        wpos4 = line[i + 4][3]
        features.extend([
            '+4:i=' + str(i + 4),

            '+4:len(syllable)=' + str(len(syllable4)),

            '+4:syllable[:1]=' + syllable4[:1],
            '+4:syllable[:2]=' + syllable4[:2],
            '+4:syllable[-1:]=' + syllable4[-1:],
            '+4:syllable[-2:]=' + syllable4[-2:],
            '+4:syllable[-3:]=' + syllable4[-3:],

            '+4:lhoc=' + lhoc4,
            '+4:lpos=' + lpos4,
            '+4:wpos=' + wpos4,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 5:
        syllable5 = line[i + 5][0]
        lhoc5 = line[i + 5][1]
        lpos5 = line[i + 5][2]
        wpos5 = line[i + 5][3]
        features.extend([
            '+5:i=' + str(i + 5),

            '+5:len(syllable)=' + str(len(syllable5)),

            '+5:syllable[:1]=' + syllable5[:1],
            '+5:syllable[:2]=' + syllable5[:2],
            '+5:syllable[-1:]=' + syllable5[-1:],
            '+5:syllable[-2:]=' + syllable5[-2:],
            '+5:syllable[-3:]=' + syllable5[-3:],

            '+5:lhoc=' + lhoc5,
            '+5:lpos=' + lpos5,
            '+5:wpos=' + wpos5,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 6:
        syllable6 = line[i + 6][0]
        lhoc6 = line[i + 6][1]
        lpos6 = line[i + 6][2]
        wpos6 = line[i + 6][3]
        features.extend([
            '+6:i=' + str(i + 6),

            '+6:len(syllable)=' + str(len(syllable6)),

            '+6:syllable[:1]=' + syllable6[:1],
            '+6:syllable[:2]=' + syllable6[:2],
            '+6:syllable[-1:]=' + syllable6[-1:],
            '+6:syllable[-2:]=' + syllable6[-2:],
            '+6:syllable[-3:]=' + syllable6[-3:],

            '+6:lhoc=' + lhoc6,
            '+6:lpos=' + lpos6,
            '+6:wpos=' + wpos6,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 7:
        syllable7 = line[i + 7][0]
        lhoc7 = line[i + 7][1]
        lpos7 = line[i + 7][2]
        wpos7 = line[i + 7][3]
        features.extend([
            '+7:i=' + str(i + 7),

            '+7:len(syllable)=' + str(len(syllable7)),

            '+7:syllable[:1]=' + syllable7[:1],
            '+7:syllable[:2]=' + syllable7[:2],
            '+7:syllable[-1:]=' + syllable7[-1:],
            '+7:syllable[-2:]=' + syllable7[-2:],
            '+7:syllable[-3:]=' + syllable7[-3:],

            '+7:lhoc=' + lhoc7,
            '+7:lpos=' + lpos7,
            '+7:wpos=' + wpos7,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 8:
        syllable8 = line[i + 8][0]
        lhoc8 = line[i + 8][1]
        lpos8 = line[i + 8][2]
        wpos8 = line[i + 8][3]
        features.extend([
            '+8:i=' + str(i + 8),

            '+8:len(syllable)=' + str(len(syllable8)),

            '+8:syllable[:1]=' + syllable8[:1],
            '+8:syllable[:2]=' + syllable8[:2],
            '+8:syllable[-1:]=' + syllable8[-1:],
            '+8:syllable[-2:]=' + syllable8[-2:],
            '+8:syllable[-3:]=' + syllable8[-3:],

            '+8:lhoc=' + lhoc8,
            '+8:lpos=' + lpos8,
            '+8:wpos=' + wpos8,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 9:
        syllable9 = line[i + 9][0]
        lhoc9 = line[i + 9][1]
        lpos9 = line[i + 9][2]
        wpos9 = line[i + 9][3]
        features.extend([
            '+9:i=' + str(i + 9),

            '+9:len(syllable)=' + str(len(syllable9)),

            '+9:syllable[:1]=' + syllable9[:1],
            '+9:syllable[:2]=' + syllable9[:2],
            '+9:syllable[-1:]=' + syllable9[-1:],
            '+9:syllable[-2:]=' + syllable9[-2:],
            '+9:syllable[-3:]=' + syllable9[-3:],

            '+9:lhoc=' + lhoc9,
            '+9:lpos=' + lpos9,
            '+9:wpos=' + wpos9,
        ])
    else:
        features.append('EOL')

    if i < len(line) - 10:
        syllable10 = line[i + 10][0]
        lhoc10 = line[i + 10][1]
        lpos10 = line[i + 10][2]
        wpos10 = line[i + 10][3]
        features.extend([
            '+10:i=' + str(i + 10),

            '+10:len(syllable)=' + str(len(syllable10)),

            '+10:syllable[:1]=' + syllable10[:1],
            '+10:syllable[:2]=' + syllable10[:2],
            '+10:syllable[-1:]=' + syllable10[-1:],
            '+10:syllable[-2:]=' + syllable10[-2:],
            '+10:syllable[-3:]=' + syllable10[-3:],

            '+10:lhoc=' + lhoc10,
            '+10:lpos=' + lpos10,
            '+10:wpos=' + wpos10,
        ])
    else:
        features.append('EOL')

    return features
