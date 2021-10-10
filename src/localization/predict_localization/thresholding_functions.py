import numpy as np
from skimage.filters import threshold_otsu


def threshold_with_float(predictions, threshold):
    """
    This function threshold and reshape all results with a fix threshold.
    @type predictions: Numpy array
    @param predictions: The trained predicted results
    @type threshold: float
    @param threshold: The fix threshold for the predictions

    @rtype: Numpy array
    @return: The thresholded predictions
    """

    return np.where(predictions > threshold, 1, 0)


def threshold_adaptive(predictions):
    """
    This function threshold and reshape all results with an adaptive threshold.
    @type predictions: Numpy array
    @param predictions: The trained predicted results

    @rtype: Numpy array
    @return: The thresholded predictions
    """

    threshold = threshold_otsu(predictions)
    return np.where(predictions > threshold, 1, 0)
