from tensorflow import keras


def fit_model(model, train_generator, validation_generator, num_epo, class_weights, early_stopping):
    """
    Train a given model with pre-processed data
    @type model: Keras model
    @param model: The model we want to train
    @type train_generator: DataFrameIterator
    @param train_generator: All the data needed for training
    @type validation_generator: DataFrameIterator
    @param validation_generator: All the data needed for validation
    @type num_epo: int
    @param num_epo: Number of epochs for training phase
    @type class_weights: None | (float, float) | float | "balanced"
    @param class_weights: Class weights. Either None for equal weights, (pos_class_weight, neg_class_weight), pos_class_weight (then neg_class_weight = 1-pos_class_weight) or balanced for automatic computation
    @type early_stopping: boolean
    @param early_stopping: Stop training if validation loss doesn't change

    @rtype: (Keras model, Keras model history)
    @return: The model, the history of the training
    """

    if early_stopping:
        callback = [keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=3, mode='auto')]
    else:
        callback = None

    history = model.fit(x=train_generator,
                        steps_per_epoch=None,
                        validation_data=validation_generator,
                        validation_steps=None,
                        epochs=num_epo,
                        class_weight=class_weights,
                        callbacks=callback)

    return model, history
