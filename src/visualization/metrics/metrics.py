from sklearn.metrics import confusion_matrix, fbeta_score


def confusion_matrix_all_parameters(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def false_positive_rate(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix_all_parameters(y_true, y_pred)
    return fp / (fp + tn)


def false_negative_rate(y_true, y_pred):
    _, _, fn, tp = confusion_matrix_all_parameters(y_true, y_pred)
    return fn / (fn + tp)


def true_negative_rate(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix_all_parameters(y_true, y_pred)
    return tn / (tn + fp)


# shouldn’t use accuracy on imbalanced problems
def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_all_parameters(y_true, y_pred)
    return (tn + tp) / (tn + fp + fn + tp)


def f_one_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)
