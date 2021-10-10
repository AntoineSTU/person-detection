from .model_architectures.build_model_from_pooling import build_model_from_pooling
from .model_architectures.build_model_from_flatten import build_model_from_flatten


def build_model(detection_model, optimizer="adam", loss="binary_crossentropy", model_type="pooling"):
    """
    This function builds up the prediction model from the trained detection model with pooling layer.
    @type detection_model: Keras model
    @param detection_model: The trained detection model
    @type optimizer: string
    @param optimizer: The optimizer used to train the model
    @default "adam"
    @type loss: string
    @param loss: The loss used to train the model
    @default "binary_crossentropy"
    @type model_type: string
    @param model_type: The type of the model we want to train (e.g. pooling, flatten...)
    @default "pooling"

    @rtype: Keras model
    @return: The localization model built from the trained detection model.
    """

    if model_type == "pooling":
        prediction_model = build_model_from_pooling(detection_model)
    elif model_type == "flatten":
        prediction_model = build_model_from_flatten(detection_model)

    # Compile
    prediction_model.compile(optimizer=optimizer, loss=loss)

    return prediction_model
