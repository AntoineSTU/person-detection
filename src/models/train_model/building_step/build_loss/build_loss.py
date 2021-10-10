from .weighted_binary_crossentropy import weighted_binary_crossentropy


def build_loss(loss_function="binary_crossentropy", class_weights=None):
    """
    Select which loss function to use in compiling the model
    @type loss_function: string
    @param loss_function: which loss function to take
    @default loss_function: binary_crossentropy
    @type class_weight: (float, float)
    @param class_weights: Corresponds to (pos_class_weight, neg_class_weight)
    @default class_weights: None

    @rtype string | function
    @return The loss function (or name) to import in Keras models
    """
    if loss_function == "weighted_binary_crossentropy":
        return weighted_binary_crossentropy(class_weights)
    else:
        return loss_function
