from tensorflow.keras import backend as K


def weighted_binary_crossentropy(class_weights):
    """
    Implement weihgted binary crossentropy for unbalanced data training
    @type class_weight: (float, float)
    @param class_weights: Corresponds to (pos_class_weight, neg_class_weight)

    @rtype: lambda (y_true: float, y_pred: float): float
    @return: weighted_binary_crossentropy
    """

    def weighted_binary_crossentropy(y_true, y_pred):

        (one_weight, zero_weight) = class_weights
        weights = y_true * one_weight + (1 - y_true) * zero_weight
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce

    return weighted_binary_crossentropy
