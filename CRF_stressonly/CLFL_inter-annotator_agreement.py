# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# Created at UC Berkeley 2015
# Authors: Christopher Hench
# ==============================================================================
'''This code presents the inter-annotator agreement matrix and
Kappa coefficient for two annotators of MHG scansion based
on the paper presented at the NAACL-CLFL 2016 by Christopher Hench
and Alex Estes. This model is for tuning.'''

import logging
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, SCORERS
import codecs
import nltk
import itertools
from nltk.tag.util import str2tuple


with open("Data/CLFL_Annotator_1_IAA.txt", "r") as f:
    a1 = f.read()

with open("Data/CLFL_Annotator_2_IAA.txt", "r") as f:
    a2 = f.read()


lines = a1.split('\n')

a_tags = [[str2tuple(x) for x in line.split()] for line in lines]
a_tags = [[x[1] for x in line if x[0] != "BEGL" and x[
    0] != "ENDL" and x[0] != "WBY"] for line in a_tags]
a_tags = [item for sublist in a_tags for item in sublist]


lines = a2.split('\n')

h_tags = [[str2tuple(x) for x in line.split()] for line in lines]
h_tags = [[x[1] for x in line if x[0] != "BEGL" and x[
    0] != "ENDL" and x[0] != "WBY"] for line in h_tags]
h_tags = [item for sublist in h_tags for item in sublist]


cm = nltk.ConfusionMatrix(a_tags, h_tags)
print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


# from sklearn
def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted
                             as equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        logger.error("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    print(observed)
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k


a_new = []
h_new = []

for tag in a_tags:

    if tag == "MORA":
        a_new.append(1)

    if tag == "MORA_HAUPT":
        a_new.append(2)

    if tag == "MORA_NEBEN":
        a_new.append(3)

    if tag == "HALB":
        a_new.append(4)

    if tag == "HALB_HAUPT":
        a_new.append(5)

    if tag == "EL":
        a_new.append(6)

    if tag == "HALB_NEBEN":
        a_new.append(7)

    if tag == "DOPPEL":
        a_new.append(8)

for tag in h_tags:
    if tag == "MORA":

        h_new.append(1)
    if tag == "MORA_HAUPT":

        h_new.append(2)
    if tag == "MORA_NEBEN":

        h_new.append(3)
    if tag == "HALB":

        h_new.append(4)
    if tag == "HALB_HAUPT":

        h_new.append(5)
    if tag == "EL":

        h_new.append(6)
    if tag == "HALB_NEBEN":

        h_new.append(7)
    if tag == "DOPPEL":

        h_new.append(8)


print("Kappa Coefficient = " + str(kappa(a_new, h_new)))
