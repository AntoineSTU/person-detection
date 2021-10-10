from tensorflow import keras

from .pre_processing import preprocess_images
from .models import predict_class
from .localization import build_localization_model, predict_localization, generate_localization_images


def run_trained_model(image_paths,
                      detection_model,
                      localization_model,
                      real_classes=None,
                      img_path_col="Image Path",
                      final_size=(600, 700),
                      preprocess_function="InceptionV3",
                      threshold_detection=0.5,
                      threshold_localization=0.5,
                      visualization_function="simple_visualization",
                      color_coeff=0.5,
                      nb_images_per_row=3,
                      image_save_path="../figures/images_with_localization.png",
                      show_picture=False):
    """
    The goal of this function is to predict localization on input images.
    This function's goal is to preprocess and load given images (/!\ don't give too much images)
    @type image_paths: string[] | string | Pandas dataframe
    @param image_paths: The image path(s)
    @type detection_model: Keras model || string
    @param detection_model: The model we want to use for the predictions people/not people or its path
    @type localization_model: Keras model || string
    @param localization_model: The model we want to use for the localization predictions or its path. You can also put the model_type if it is not yet constructed
    @type real_classes: None | int list
    @param real_classes: The real classes of the images
    @default None
    @type img_path_col: None | string
    @param img_path_col: The column name containing the image paths if image_paths is a dataframe
    @default "Image Path"
    @type final_size: (int, int)
    @param final_size: The size our training images should have
    @default (600, 700)
    @type preprocess_function: string | (image: Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @default "InceptionV3"
    @type threshold_detection: float || string
    @param threshold_detection: The threshold to apply on the class predictions ("none" for no threshold, "adaptive" for adaptive threshold)
    @default 0.5
    @type threshold_localization: float || string
    @param threshold_localization: The threshold to apply on the localization predictions ("none" for no threshold, "adaptive" for adaptive threshold)
    @default 0.5
    @type visualization_function: string
    @param visualization_function: The function tu use to show vizualisation on each image
    @default "simple_visualization"
    @type color_coeff: Float between 0 and 1
    @param color_coeff: How much the areas with people will be colored 
    @default 0.5
    @type nb_images_per_row: Integer
    @param nb_images_per_row: How much images per row
    @default 4
    @type image_save_path: string
    @param image_save_path: Where to save the image (nowhere if None)
    @default "../figures/images_with_localization.png"
    @type show_picture: boolean
    @param show_picture: If we should show the images or not
    @default False

    @rtype: {"classes": [Numpy array], "localizations": [Numpy array]}
    @return: The predictions
    """

    # First we preprocess the images
    print("Beginning preprocessing")
    all_images = preprocess_images(
        image_paths, img_path_col, final_size, preprocess_function, True)
    loaded_images = all_images["loaded_images"]
    preprocessed_images = all_images["preprocessed_images"]
    print("Preprocessing ended")

    # Then we load the detection model if we got a path
    if isinstance(detection_model, str):
        print("Loading the detection model")
        detection_model = keras.models.load_model(detection_model)
        print("Detection model loaded")

    # We can generate the classes
    print("Getting the classes")
    classes = predict_class(
        detection_model, preprocessed_images, threshold_detection)
    print("Classes generated")

    # Then we load the localization model if we got a path
    if isinstance(localization_model, str):
        if "/" in localization_model:
            print("Loading the localization model")
            localization_model = keras.models.load_model(localization_model)
            print("Localization model loaded")
        else:
            print("Building the localization model")
            localization_model = build_localization_model(
                detection_model, model_type=localization_model)
            print("Localization model built")

    # We make up the image titles
    if real_classes != None:
        img_titles = ["Real {}; predicted {}".format(
            real_classes[i], classes[i][0]) for i in range(len(preprocessed_images))]
    else:
        img_titles = ["Predicted {}".format(
            classes[i][0]) for i in range(len(preprocessed_images))]

    # We can then get the localizations
    print("Getting the localizations")
    localizations = predict_localization(
        localization_model, preprocessed_images, threshold_localization)
    print("Localizations generated")

    # We can finally produce the final image
    print("Generating localization images")
    generate_localization_images(
        loaded_images, localizations, visualization_function, img_titles, color_coeff, nb_images_per_row, image_save_path, show_picture)

    return {"classes": classes, "localizations": localizations}
