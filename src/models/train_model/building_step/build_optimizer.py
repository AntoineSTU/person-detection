from tensorflow.keras import optimizers


def build_optimizer(opt="adam", learning_rate=0.001):
    """
    Select optimizer with a learning rate to use in compiling the model
    @type opt: string
    @param opt: Which optimizer function to take
    @default opt: adam
    @type learning_rate: float
    @param learning_rate: Corresponding learning_rate for the optimizer
    @default learning_rate: 0.001

    @rtype Keras optimizer
    @return optimizer
    """

    if opt == "adam":
        return optimizers.Adam(learning_rate)

    if opt == "rmsprop":
        return optimizers.RMSprop(learning_rate)

    if opt == "sgd":
        return optimizers.SGD(learning_rate)

    if opt == "adadelta":
        return optimizers.Adadelta(learning_rate)
