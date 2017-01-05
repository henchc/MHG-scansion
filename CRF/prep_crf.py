import nltk
from get_features import syllableend, syllableweight


def prep_crf(taggedstring):

    taggedstring = taggedstring.replace("MORA_NEBEN", "MORA_HAUPT")
    taggedstring = taggedstring.replace("HALB_NEBEN", "HALB_HAUPT")

    # break data into list on line breaks
    tagged = taggedstring.split('\n')

    # create tuples with syllable characteristic features
    training = []
    for line in tagged:
        news = [nltk.tag.str2tuple(t)
                for t in line.split()]  # make tags into tuples
        if len(news) > 0:
            newtups = []
            for tup in news:
                # adds syllable characteristics to tuple for being feature
                syllweight = (
                    tup[0], syllableend(
                        tup[0]), syllableweight(
                        tup[0]), tup[1])
                newtups.append(syllweight)
            training.append(newtups)

    # add features for word boundary and line boundary
    ftuples = []
    for line in training:
        ltuples = []
        for i, tup in enumerate(line):
            if tup[0] == 'WBY':
                pass
            elif i == 0:
                lpos = "BEGL"
                if line[i + 1][0] == "WBY":
                    wpos = "MONO"
                else:
                    wpos = "WBYL"
                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

            elif i > 0 and i < len(line) - 1:

                lpos = "LPNA"

                if line[i - 1][0] == "WBY" and line[i + 1][0] == "WBY":
                    wpos = "MONO"
                elif line[i - 1][0] == "WBY":
                    wpos = "WBYL"
                elif line[i + 1][0] == "WBY":
                    wpos = "WBYR"
                else:
                    wpos = "WBNA"

                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

            elif i == len(line) - 1:
                lpos = "ENDL"
                if line[i - 1][0] == "WBY":
                    wpos = "MONO"
                else:
                    wpos = "WBYR"
                finaltuple = (tup[0], tup[1], tup[2], lpos, wpos, tup[3])
                ltuples.append(finaltuple)

        ftuples.append(ltuples)

    return (ftuples)
