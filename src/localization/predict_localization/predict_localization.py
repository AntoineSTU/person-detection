from .thresholding_functions import threshold_with_float, threshold_adaptive


def predict_localization(model, images, threshold=0.5):
    """
    Return predictions for given model and data
    @type model: Keras model
    @param model: The model we want to use for the predictions
    @type images: numpy array
    @param images: The preprocessed images for the prediction
    @type threshold: float || string
    @param threshold: The threshold to apply on the predictions ("none" for no threshold, "adaptive" for adaptive threshold)
    @default 0.5

    @rtype: numpy array
    @return: The predictions
    """

    predictions = model.predict(images)
    predictions = predictions.reshape(
        predictions.shape[0], predictions.shape[1], predictions.shape[2])
    if isinstance(threshold, float):
        formatted_predictions = threshold_with_float(predictions, threshold)
    elif threshold == "adaptive":
        formatted_predictions = threshold_adaptive(predictions)
    else:
        formatted_predictions = predictions
    return formatted_predictions
