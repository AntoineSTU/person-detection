from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow import keras


def build_model_with_pooling(img_h, img_w):
    """
    Build a model with a pooling layer
    @type img_h: int
    @param img_h: Height of the images (nb of pixels)
    @type img_w: int
    @param img_w: Width of the images (nb of pixels)

    @rtype: Keras model
    @return: The model
    """

    # Inception model
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_h, img_w, 3),
        pooling="avg"
    )

    # Freeze convolutional layers (are not retrained)
    base_model.trainable = False

    # Fully connected layer
    output = layers.Dense(units=1, activation='sigmoid')(
        base_model.layers[-1].output)

    # Build
    model = keras.Model(base_model.inputs, output)

    return model
