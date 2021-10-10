import numpy as np


def threshold_with_float(predictions, threshold):
    """
    This function threshold and reshape all results.
    @type predictions: Numpy array
    @param predictions: The trained predicted results
    @type threshold: float
    @param threshold: The fix threshold for the predictions

    @rtype: Numpy array
    @return: The thresholded predictions
    """

    return np.where(predictions > threshold, 1, 0)
