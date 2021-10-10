import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D


def build_model_from_pooling(detection_model):
    """
    This function builds up the prediction model from the trained detection model with pooling layer.
    @type detection_model: Keras model
    @param detection_model: The trained detection model

    @rtype: Keras model
    @return: The localization model built from the trained detection model.
    """

    # First we get the weights of the dense layer and we format them for the new Conv2D layer
    dense_layer = detection_model.layers[-1]
    dense_layer_weights = dense_layer.get_weights()
    formatted_weights = (
        np.array([[dense_layer_weights[0]]]), dense_layer_weights[1])

    # Then we create the new model be deleting the Pooling + Dense layers and adding a new Conv2D layer
    prediction_outputs = Conv2D(1, (1, 1), activation='sigmoid', weights=formatted_weights,
                                name='conv_predictor')(detection_model.layers[-3].output)
    prediction_model = Model(
        inputs=detection_model.inputs, outputs=prediction_outputs)

    return prediction_model
