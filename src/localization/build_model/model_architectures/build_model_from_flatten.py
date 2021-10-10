import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape


def build_model_from_flatten(detection_model):
    """
    This function builds up the prediction model from the trained detection model with flatten layer.
    @type detection_model: Keras model
    @param detection_model: The trained detection model

    @rtype: Keras model
    @return: The localization model built from the trained detection model.
    """

    # First we get the weights of the dense layer and we format them for the new Conv2D layer
    dense_layer = detection_model.layers[-1]
    dense_layer_weights = dense_layer.get_weights()
    _, output_height, output_width, output_depth = detection_model.layers[-3].output_shape
    flattened_weights = np.zeros(
        (output_width*output_height*output_depth, output_width*output_height))
    for i in range(output_height):
        for j in range(output_width):
            weights_i_j = np.array(dense_layer_weights[0][i*output_width*output_depth + j*output_depth: i*output_width *
                                                          output_depth + j*output_depth + output_depth].ravel())
            flattened_weights[i*output_width*output_depth + j*output_depth:i*output_width*output_depth +
                              j*output_depth + output_depth, i*output_width+j] = weights_i_j
    formatted_weights = (
        flattened_weights, dense_layer_weights[1]*np.ones(output_width*output_height))

    # Then we create the new model be deleting the Dense layer and adding new Dense + Reshape layers
    prediction_flat_outputs = Dense(output_width*output_height, activation='sigmoid',
                                    weights=formatted_weights, name='dense_predictor')(detection_model.layers[-2].output)
    prediction_outputs = Reshape(
        (output_height, output_width))(prediction_flat_outputs)
    prediction_model = Model(
        inputs=detection_model.inputs, outputs=prediction_outputs)

    return prediction_model
