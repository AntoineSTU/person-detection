import numpy as np
from sklearn.utils import class_weight
from .building_step.build_step import build_model
from .testing_step.testing_step import test_model
from .fitting_step.fitting_step import fit_model


def train_model(training_data, optimizer="adam", learning_rate=0.001, loss="binary_crossentropy", num_epo=20, model_type="pooling", class_weights=None, early_stopping=False):
    """
    Train a model with pre-processed data
    @type training_data: {train: DataFrameIterator, validation: DataFrameIterator, test: DataFrameIterator}
    @param training_data: All the data needed for training
    @type optimizer: string
    @param optimizer: The optimizer used to train the model
    @default "adam"
    @type learning rate: float
    @param learning rate: The learning rate of the optimizer
    @default 0.001
    @type loss: string
    @param loss: The loss used to train the model
    @default "binary_crossentropy"
    @type num_epo: int
    @param num_epo: Number of epochs for training phase
    @default 20    
    @type model_type: string
    @param model_type: The type of the model we want to train (e.g. pooling, flatten...)
    @default "pooling"
    @type class_weights: None | (float, float) | float | "balanced"
    @param class_weights: Class weights. Either None for equal weights, (pos_class_weight, neg_class_weight), pos_class_weight (then neg_class_weight = 1-pos_class_weight) or balanced for automatic computation
    @default None
    @type early_stopping: boolean
    @param early_stopping: Stop training if validation loss doesn't change
    @default False

    @rtype: {model: Keras model, test_predictions: numpy array, test_metrics: dict of metrics}
    @return: The model, the predictions, the metrics
    """

    train_generator = training_data["train"]
    validation_generator = training_data["validation"]
    test_generator = training_data["test"]

    (img_h, img_w) = train_generator.target_size

    # Format class weights
    formatted_class_weights = class_weights
    if isinstance(class_weights, float):
        formatted_class_weights = (class_weights, 1-class_weights)
    elif class_weights == "balanced":
        if loss == "weighted_binary_crossentropy":
            positive_class_weight = train_generator.classes.count(
                0)/len(train_generator.classes)
            formatted_class_weights = (
                positive_class_weight, 1-positive_class_weight)
        else:
            formatted_class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    elif class_weights == None and loss == "weighted_binary_crossentropy":
        formatted_class_weights = (0.5, 0.5)
    print("Computed class weights: {}".format(formatted_class_weights))

    # First we build the model
    model = build_model(img_h, img_w, optimizer,
                        learning_rate, loss, formatted_class_weights, model_type)

    # Then we train it
    global_class_weights = None if loss == "weighted_binary_crossentropy" else formatted_class_weights
    model, history = fit_model(model, train_generator, validation_generator,
                               num_epo, global_class_weights, early_stopping)

    # Finally we test it
    results_test = test_model(model, test_generator)

    return {"model": model, "test_predictions": results_test[0], "test_metrics": {"accuracy": results_test[1]}, "history": history}
