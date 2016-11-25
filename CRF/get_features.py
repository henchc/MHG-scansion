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


def get_features(lines_sylls):

    text_with_features = []
    for line in lines_sylls:
        line_with_features = []
        for i, word in enumerate(line):
            for i2, syll in enumerate(word):

                # get line position of syllable
                if i == 0 and i2 == 0:
                    lpos = "BEGL"
                elif i == len(line) - 1 and i2 == len(word) - 1:
                    lpos = "ENDL"
                else:
                    lpos = "LPNA"

                # get word position of syllable
                if len(word) == 1:
                    wpos = "MONO"
                elif i2 == 0:
                    wpos = "WBYL"
                elif i2 == len(word) - 1:
                    wpos = "WBYR"
                else:
                    wpos = "WBNA"

                syll_features = (syll, syllableweight(
                    syll) + syllableend(syll), lpos, wpos)

                line_with_features.append(syll_features)
        text_with_features.append(line_with_features)

    # clear empty lists
    text_with_features = [x for x in text_with_features if len(x) > 0]

    return text_with_features
