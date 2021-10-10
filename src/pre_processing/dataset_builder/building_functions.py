import pandas as pd
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_dataset(image_df, img_class_col, nb_images, is_balanced):
    """
    This function's goal is to extract from the dataframe the images we will use while training our model
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type img_class_col: string
    @param img_class_col: The column name from which we can extract the classes (0/1)
    @type nb_images: int
    @param nb_images: The nb of images we will use to train our model
    @type is_balanced: boolean
    @param is_balanced: In the dataset should be balanced or not

    @rtype: Pandas dataframe
    @return: The filtered dataframe
    """

    image_df = image_df.sample(frac=1).reset_index(drop=True)
    if nb_images == -1:
        nb_images = image_df.shape[0]
    if is_balanced:
        df_not_people = image_df[image_df[img_class_col] == "not_people"]
        df_people = image_df[image_df[img_class_col] == "people"]
        nb_images_from_each = min(
            nb_images//2, df_not_people.shape[0], df_people.shape[0])
        final_df = pd.concat(
            [df_not_people.iloc[:nb_images_from_each], df_people.iloc[:nb_images_from_each]])
        return final_df
    else:
        return image_df.iloc[:min(nb_images, image_df.shape[0])]


def load_dataset(image_df, img_path_col, dataset_type_col, img_class_col, final_size, preprocess_function, batch_size):
    """
    This function's goal is to load the images we will use while training our model
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type img_path_col: string
    @param img_path_col: The column name containing the image paths
    @type dataset_type_col: string
    @param dataset_type_col: The column name containing the dataset types (train/validation/test)
    @type img_class_col: string
    @param img_class_col: The column name from which we can extract the classes (people/not_people)
    @type final_size: (int, int)
    @param final_size: The size our training images should have
    @type preprocess_function: string | (image: Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @type batch_size: int
    @param batch_size: Number of images by batch

    @rtype: {train: DataFrameIterator, validation: DataFrameIterator, test: DataFrameIterator}
    @return: All the data needed for training
    """

    if preprocess_function == "InceptionV3":
        preprocess_function = preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function
    )

    train_generator = train_datagen.flow_from_dataframe(
        image_df[image_df[dataset_type_col] == "train"],
        directory=None,
        x_col=img_path_col,
        y_col=img_class_col,
        target_size=final_size,
        classes={"people": 1, "not_people": 0},
        class_mode="binary",
        batch_size=batch_size)

    validation_generator = train_datagen.flow_from_dataframe(
        image_df[image_df[dataset_type_col] == "validation"],
        directory=None,
        x_col=img_path_col,
        y_col=img_class_col,
        target_size=final_size,
        classes={"people": 1, "not_people": 0},
        class_mode="binary",
        batch_size=batch_size)

    test_generator = train_datagen.flow_from_dataframe(
        image_df[image_df[dataset_type_col] == "test"],
        directory=None,
        x_col=img_path_col,
        y_col=img_class_col,
        target_size=final_size,
        classes={"people": 1, "not_people": 0},
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False)

    return {
        "train": train_generator,
        "validation": validation_generator,
        "test": test_generator
    }
