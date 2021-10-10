from .cleaning_step.cleaning_step import cleaning_step
from .dataset_builder.dataset_builder import dataset_builder
from .dataset_formatter.dataset_formatter import dataset_formatter
from .simple_preprocessing.image_preprocessing import preprocess_images


def default_class_format(keywords):
    return "people" if "people" in keywords else "not_people"


def default_path_function(image_name, directory_path):
    return directory_path + '/' + image_name + '.TIF'


def create_formatted_dataset(image_df,
                             extract_classes=True,
                             extract_img_path=True,
                             extract_shape=True,
                             class_column="Keywords",
                             img_name_col="File",
                             image_path_function=default_path_function,
                             class_format_function=default_class_format,
                             train_dir_path="/original/RAISE/RaiseTrain",
                             valid_dir_path="/original/RAISE/RaiseVal",
                             test_dir_path="/original/RAISE/RaiseTest",
                             final_class_col_name="Class",
                             final_type_col_name="Dataset Type",
                             final_path_col_name="Image Path",
                             final_height_col_name="Image Height",
                             final_width_col_name="Image Width"):
    """
    This function implement the first steps of the dataset formatting (that we need to execute 
    every time with the same parameters): it extract the classes and add image paths.
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type extract_classes: boolean
    @param extract_classes: If we should extract classes or not
    @default True
    @type extract_img_path: boolean
    @param extract_img_path: If we should extract paths from train/validation/test folders or not
    @default True
    @type extract_shape: boolean
    @param extract_img_path: If we should extract image shapes or not
    @default True
    @type class_column: string
    @param class_column: The column name from which we can extract the classes
    @default "Keywords"
    @type img_name_col: string
    @param img_name_col: The column name containing the image names
    @default "File"
    @type image_path_function: (image_name: any, directory_path: string) -> string
    @param image_path_function: A function returning image paths based on their names (cf. name col) and the directory path
    @default (image_name: string, directory_path: string) -> directory_path + '/' + image_name + '.TIF'
    @type class_format_function: (class: any) -> 0 | 1
    @param class_format_function: A function that takes object from the class column and that returns 0 (not people) or 1 (people)
    @default (keywords: string[]) -> float('people' in keywords)
    @type train_dir_path: string
    @param train_dir_path: The path of the train directory
    @default "/original/RAISE/RaiseTrain"
    @type valid_dir_path: string
    @param valid_dir_path: The path of the validation directory
    @default "/original/RAISE/RaiseVal"
    @type test_dir_path: string
    @param test_dir_path: The path of the test directory
    @default "/original/RAISE/RaiseTest"
    @param final_class_col_name: string
    @type final_class_col_name: The name of the final class column
    @default "Class"
    @param final_type_col_name: string
    @type final_type_col_name: The name of the final dataset type column
    @default "Dataset Type"
    @param final_path_col_name: string
    @type final_path_col_name: The name of the final image path column
    @default "Image Path"
    @param final_height_col_name: string
    @type final_height_col_name: The name of the final image height column
    @default "Image Height"
    @param final_width_col_name: string
    @type final_width_col_name: The name of the final image width column
    @default "Image Width"

    @rtype: Pandas dataframe
    @return: The same as before with the class ("Class"), dataset type ("Dataset Type") and image path ("Image Path") columns added
    """

    return dataset_formatter(image_df, extract_classes, extract_img_path, extract_shape, class_column, img_name_col,
                             image_path_function, class_format_function, train_dir_path, valid_dir_path,
                             test_dir_path, final_class_col_name, final_type_col_name, final_path_col_name,
                             final_height_col_name, final_width_col_name)


def pre_processing_train(image_df,
                         img_path_col="Image Path",
                         class_col="Class",
                         dataset_type_col="Dataset Type",
                         image_height_col="Image Height",
                         image_width_col="Image Width",
                         filter_with_size=True,
                         filter_with_size_parameters={
                             "sizes_to_filter": [(3008, 2000), (2000, 3008)]},
                         filter_with_mode=True,
                         filter_with_mode_parameters={
                             "portrait_mode_filtered": True},
                         nb_images=-1,
                         is_balanced=False,
                         final_size=(600, 700),
                         preprocess_function="InceptionV3",
                         batch_size=50):
    """
    This function's goal is to prepare the dataset and load the images.
    on its mode (portrait mode or landscape mode)
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
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
    @param final_size: The size our training images should have (height, width)
    @default (600, 700)
    @type preprocess_function: string | (Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @default InceptionV3
    @type batch_size: int
    @param batch_size: Number of images by batch
    @default 50

    @rtype: {train: DataFrameIterator, validation: DataFrameIterator, test: DataFrameIterator}
    @return: All the data needed for training
    """

    print("Pre-processing: before filtering")
    filterd_df = cleaning_step(image_df, image_height_col, image_width_col,
                               filter_with_size, filter_with_size_parameters,
                               filter_with_mode, filter_with_mode_parameters)
    print("Pre-processing: after filtering")
    loaded_dataset = dataset_builder(filterd_df, img_path_col, dataset_type_col, class_col, nb_images,
                                     is_balanced, final_size, preprocess_function, batch_size)
    print("Pre-processing: dataset loaded")
    return loaded_dataset
