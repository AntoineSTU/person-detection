from .thresholding_functions import threshold_with_float


def predict_class(model, images, threshold=0.5):
    """
    Return predictions for given model and data
    @type model: Keras model
    @param model: The model we want to use for the predictions
    @type images: numpy array
    @param images: The preprocessed images for the prediction
    @type threshold: float || string
    @param threshold: The threshold to apply on the predictions ("none" for no threshold)
    @default 0.5

    @rtype: numpy array
    @return: The predictions
    """

    predictions = model.predict(images)
    if isinstance(threshold, float):
        formatted_predictions = threshold_with_float(predictions, threshold)
    else:
        formatted_predictions = predictions
    return formatted_predictions
