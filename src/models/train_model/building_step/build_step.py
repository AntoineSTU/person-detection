from .model_architecture.build_step_with_pooling import build_model_with_pooling
from .model_architecture.build_step_with_flatten import build_model_with_flatten
from .build_optimizer import build_optimizer
from .build_loss.build_loss import build_loss
import logging


def build_model(img_h, img_w, optimizer, learning_rate, loss, class_weights, model_type):
    """
    Build the model (but do not train it)
    @type img_h: int
    @param img_h: Image height (nb of pixels)
    @type img_w: int
    @param img_w: Image width (nb of pixels)
    @type learning_rate: float
    @param learning_rate: Corresponding learning_rate for the optimizer
    @type loss: string
    @param loss: The loss used to train the model
    @type class_weights: None | (float, float) | float | "balanced"
    @param class_weights: Class weights. Either None for equal weights, (pos_class_weight, neg_class_weight), pos_class_weight (then neg_class_weight = 1-pos_class_weight) or balanced for automatic computation
    @type model_type: string
    @param model_type: The type of the model we want to train (e.g. pooling, flatten...)

    @rtype: Keras model
    @return: The model, to be trained
    """

    logging.info('Building Model')

    if model_type == "pooling":
        model = build_model_with_pooling(img_h, img_w)
    elif model_type == "flatten":
        model = build_model_with_flatten(img_h, img_w)

    # Compile
    optimizer = build_optimizer(optimizer, learning_rate)
    loss = build_loss(loss, class_weights)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print("Model structure: ", model.summary())

    return model
