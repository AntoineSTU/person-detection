import numpy as np
import pandas as pd
import logging
from datetime import datetime

from .pre_processing import pre_processing_train
from .models import train_model
from .visualization import visualize
from .localization import build_localization_model


def train_model_from_scratch(image_df_path,
                             img_path_col="Image Path",
                             class_col="Class",
                             dataset_type_col="Dataset Type",
                             image_height_col="Image Height",
                             image_width_col="Image Width",
                             filter_with_size=True,
                             filter_with_size_parameters={
                                 "sizes_to_filter": [(3008, 2000), (2000, 3008)]
                             },
                             filter_with_mode=True,
                             filter_with_mode_parameters={
                                 "portrait_mode_filtered": True
                             },
                             nb_images=-1,
                             is_balanced=False,
                             final_size=(600, 700),
                             preprocess_function="InceptionV3",
                             model_type="pooling",
                             optimizer="adam",
                             learning_rate=0.001,
                             loss="binary_crossentropy",
                             batch_size=50,
                             num_epo=20,
                             class_weights=None,
                             early_stopping=False,
                             figures_save_path="../reports/figures",
                             model_params_save_path="../reports/run_parameters/parameters.txt",
                             detection_model_save_path="../models/detection_model",
                             localization_model_save_path="../models/localization_model"):
    """
    The goal of this function is to train a model from scratch.
    @type image_df_path: string
    @param image_df: The path to the dataframe containing the images as a csv file
    @type img_path_col: string
    @param img_path_col: The column name containing the image paths
    @default "Image Path"
    @type class_col: string
    @param class_col: The column name from which we can extract the classes (0/1)
    @default "Class"
    @type dataset_type_col: string
    @param dataset_type_col: The column name containing the dataset types (train/validation/test)
    @default "Dataset Type"
    @type image_height_col: string
    @param image_height_col: The col name for image height
    @default "Image Height"
    @type image_width_col: string
    @param image_width_col: The col name for image width
    @default "Image Width"
    @type filter_with_size: boolean
    @param filter_with_size: If the images should be filtered based on their size
    @default True
    @type filter_with_size_parameters: None | {"sizes_to_filter": (int, int)[]}
    @param filter_with_size_parameters: All the sizes to filter (None if this step is skipped)
    @default {"sizes_to_filter": [(3008, 2000), (2000, 3008)]}
    @type filter_with_mode: boolean
    @param filter_with_mode: If the images should be filtered based on their mode (portrait/landscape)
    @default True
    @type filter_with_mode_parameters: None | {"portrait_mode_filtered": boolean}
    @param filter_with_mode_parameters: If the landscape mode images should be filtered (else the landcape will) (None if this step is skipped)
    @default {"portrait_mode_filtered": True}
    @type nb_images: int
    @param nb_images: The nb of images we will use to train our model or -1 for the whole dataset
    @default -1
    @type is_balanced: boolean
    @param is_balanced: In the dataset should be balanced or not
    @default True
    @type final_size: (int, int)
    @param final_size: The size our training images should have
    @default (600, 700)
    @type preprocess_function: string | (Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @default InceptionV3
    @type model_type: string
    @param model_type: The type of the model we want to train (e.g. pooling, flatten...)
    @default "pooling"
    @type optimizer: string
    @param optimizer: The optimizer used to train the model
    @default "adam"
    @type learning rate: float
    @param learning rate: The learning rate of the optimizer
    @default 0.001
    @type loss: string
    @param loss: The loss used to train the model
    @default "binary_crossentropy"
    @type batch_size: int
    @param batch_size: Number of images by batch
    @default 50
    @type num_epo: int
    @param num_epo: Number of epochs for training phase
    @default 20
    @type class_weights: None | (float, float) | float | "balanced"
    @param class_weights: Class weights. Either None for equal weights, (pos_class_weight, neg_class_weight), pos_class_weight (then neg_class_weight = 1-pos_class_weight) or balanced for automatic computation
    @default None
    @type early_stopping: boolean
    @param early_stopping: Stop training if validation loss doesn't change
    @default False
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures
    @default "../reports/figures"
    @type model_params_save_path: string
    @param model_params_save_path: The path where to save the model parameters
    @default "../reports/run_parameters/parameters.txt"
    @type detection_model_save_path: string
    @param detection_model_save_path: The path where to save the detection model
    @default "../models/detection_model"
    @type localization_model_save_path: string
    @param localization_model_save_path: The path where to save the localization model
    @default "../models/localization_model"

    @rtype: {"detection_model": Keras model, "localization_model": Keras model, "metrics": dictionary {string: float}}
    @return: The detection model, the localization model and the metrics produced on the test set
    """

    # We will use that as an id to store the run parameters and link them with the model

    image_df = pd.read_csv(image_df_path)

    # First we preprocess the data
    print("Beginning preprocessing")
    logging.info("Beginning preprocessing")
    preprocessed_data = pre_processing_train(image_df,
                                             img_path_col,
                                             class_col,
                                             dataset_type_col,
                                             image_height_col,
                                             image_width_col,
                                             filter_with_size,
                                             filter_with_size_parameters,
                                             filter_with_mode,
                                             filter_with_mode_parameters,
                                             nb_images,
                                             is_balanced,
                                             final_size,
                                             preprocess_function,
                                             batch_size)
    test_true_labels = np.array(
        preprocessed_data["test"].classes).reshape(-1, 1)
    print("Preprocessing ended")
    logging.info("Preprocessing ended")

    # Then we build and train the model
    result_train = train_model(
        preprocessed_data, optimizer, learning_rate, loss, num_epo, model_type, class_weights, early_stopping)
    detection_model = result_train["model"]
    test_predictions = result_train["test_predictions"]
    print("Train/test ended")
    logging.info("Train/test ended")

    # We save the model and its params
    parameters_str = """
    Model trained on the {}.

    Filtering params:

    filter_with_size: {}
    filter_with_size_parameters: {}
    filter_with_mode: {}
    filter_with_mode_parameters: {}

    Preprocessing params:

    is_balanced: {}
    final_size: {}
    preprocess_function: {}

    Model build params:

    model_type: {}
    optimizer: {}
    learning_rate: {}
    loss: {}
    class_weights: {}

    Model run params:

    nb_images: {}
    batch_size: {}
    num_epo: {}

    Results:

    Accuracy: {} on {} test images
    """.format(
        datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        filter_with_size,
        filter_with_size_parameters,
        filter_with_mode,
        filter_with_mode_parameters,
        is_balanced,
        final_size,
        preprocess_function,
        model_type,
        optimizer,
        learning_rate,
        loss,
        class_weights,
        preprocessed_data["train"].samples,
        batch_size,
        num_epo,
        result_train["test_metrics"]["accuracy"],
        preprocessed_data["test"].samples
    )
    with open(model_params_save_path, "w") as text_file:
        text_file.write(parameters_str)

    # We save the model
    detection_model.save(detection_model_save_path)
    print("Detection model saved")
    logging.info("Detection model saved")

    # Finally we produce some graphs/metrics about the model
    metrics = visualize(test_true_labels, test_predictions,
                        result_train["history"], figures_save_path)
    print("Graphs saved")
    logging.info("Graphs saved")

    # Now we can create the prediction model. We don't need the optimizer and the loss as we don't train it
    localization_model = build_localization_model(
        detection_model, model_type=model_type)
    localization_model.save(localization_model_save_path)
    print("Localization model saved")
    logging.info("Localization model saved")

    return {"detection_model": detection_model, "localization_model": localization_model, "metrics": metrics}
