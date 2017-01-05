# designate features for model to collect
def syllable2features(line, i):
    syllable = line[i][0]
    oc = line[i][1]  # light, heavy, open, closed
    lh = line[i][2]
    lpos = line[i][3]  # position in line
    wpos = line[i][4]  # position in word

    features = [
        'bias',  # bias to current syllable

        'len(syllable)=' + str(len(syllable)),  # length of syllable

        'syllable[:1]=' + syllable[:1],  # first letter
        'syllable[:2]=' + syllable[:2],  # first two letters
        'syllable[-1:]=' + syllable[-1:],  # last letter
        'syllable[-2:]=' + syllable[-2:],  # last two letters


        'oc=' + oc,
        'lh=' + lh,
        'lpos=' + lpos,
        'wpos=' + wpos,

    ]

    for p in range(1, len(line)):
        if i > p:
            syllablen1 = line[i - p][0]
            ocn1 = line[i - p][1]
            lhn1 = line[i - p][2]
            lposn1 = line[i - p][3]
            wposn1 = line[i - p][4]
            features.extend([
                '-' + str(p) + ':len(syllable)=' + str(len(syllablen1)),

                '-' + str(p) + ':syllable[:1]=' + syllablen1[:1],
                '-' + str(p) + ':syllable[:2]=' + syllablen1[:2],
                '-' + str(p) + ':syllable[-1:]=' + syllablen1[-1:],
                '-' + str(p) + ':syllable[-2:]=' + syllablen1[-2:],

                '(' + str(p) + ':syllable[-1] + syllable[0])=%s' +
                (syllablen1[-2:] + syllable[:2]),

                '-' + str(p) + ':oc=' + ocn1,
                '-' + str(p) + ':lh=' + lhn1,
                '-' + str(p) + ':lpos=' + lposn1,
                '-' + str(p) + ':wpos=' + wposn1,
            ])

        else:
            features.append('BOL')

    for p in range(1, len(line)):
        if i < len(line) - p:
            syllable1 = line[i + p][0]
            oc1 = line[i + p][1]
            lh1 = line[i + p][2]
            lpos1 = line[i + p][3]
            wpos1 = line[i + p][4]
            features.extend([
                '+' + str(p) + ':len(syllable)=' + str(len(syllable1)),

                '+' + str(p) + ':syllable[:1]=' + syllable1[:1],
                '+' + str(p) + ':syllable[:2]=' + syllable1[:2],
                '+' + str(p) + ':syllable[-1:]=' + syllable1[-1:],
                '+' + str(p) + ':syllable[-2:]=' + syllable1[-2:],

                '(-' + str(p) + ':syllable[-1] + syllable[0])=%s' +
                (syllable[-2:] + syllable1[:2]),

                '+' + str(p) + ':oc=' + oc1,
                '+' + str(p) + ':lh=' + lh1,
                '+' + str(p) + ':lpos=' + lpos1,
                '+' + str(p) + ':wpos=' + wpos1,

            ])

        else:
            features.append('EOL')

    return features


def line2features(line):
    return [syllable2features(line, i) for i in range(len(line))]


def line2labels(line):
    return [label for token, oc, lh, lpos1, wpos1, label in line]


def line2tokens(line):
    return [token for token, oc, lh, lpos1, wpos1, label in line]
