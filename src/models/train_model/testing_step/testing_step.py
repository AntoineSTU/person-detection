from tensorflow import keras


def test_model(model, test_generator):
    """
    To test the model on a test dataset
    @type model: Keras model
    @param model: The trained model to tested
    @type test_generator: DataFrameIterator
    @param training_data: All the data needed for testing

    @rtype: (Numpy array, string)
    @return: The predicted labels, the accuracy
    """

    prediction = model.predict(test_generator)
    _, accuracy = model.evaluate(test_generator)

    print("Test accuracy: {}".format(accuracy))

    return prediction, accuracy
