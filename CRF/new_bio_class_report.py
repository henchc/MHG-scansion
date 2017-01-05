from CLFL_mdf_classification import classification_report
from CLFL_mdf_classification import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from itertools import chain


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from
    github master) to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    labs = [class_indices[cls] for cls in tagset]

    return((precision_recall_fscore_support(y_true_combined,
                                            y_pred_combined,
                                            labels=labs,
                                            average=None,
                                            sample_weight=None)),
           (classification_report(
               y_true_combined,
               y_pred_combined,
               labels=[class_indices[cls] for cls in tagset],
               target_names=tagset,
           )), labs)
