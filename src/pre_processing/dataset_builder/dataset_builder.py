from .building_functions import extract_dataset, load_dataset


def dataset_builder(image_df, img_path_col="Image Path", dataset_type_col="Dataset Type", class_column="Class", nb_images=-1,
                    is_balanced=True, final_size=(600, 700), preprocess_function="InceptionV3", batch_size=50):
    """
    This function's goal is to build the dataset and to import images to numpy arrays
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type img_path_col: string
    @param img_path_col: The column name containing the image paths
    @default "Image Path"
    @type dataset_type_col: string
    @param dataset_type_col: The column name containing the dataset types (train/validation/test)
    @default "Dataset Type"
    @type class_column: string
    @param class_column: The column name from which we can extract the classes (0/1)
    @default "Class"
    @type nb_images: int
    @param nb_images: The nb of images we will use to train our model or -1 for the whole dataset
    @default -1
    @type is_balanced: boolean
    @param is_balanced: In the dataset should be balanced or not
    @default True
    @type final_size: (int, int)
    @param final_size: The size our training images should have
    @default (600, 700)
    @type preprocess_function: string | (image: Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @default InceptionV3
    @type batch_size: int
    @param batch_size: Number of images by batch
    @default 50

    @rtype: {train: DataFrameIterator, validation: DataFrameIterator, test: DataFrameIterator}
    @return: All the data needed for training
    """

    final_df = extract_dataset(
        image_df, class_column, nb_images, is_balanced)
    return load_dataset(final_df, img_path_col, dataset_type_col, class_column, final_size, preprocess_function, batch_size)
