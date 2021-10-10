from .class_formatting_functions import format_class_names
from .image_retriever_functions import extract_train_valid_test
from .shape_retriever_functions import extract_height_width


def dataset_formatter(image_df,
                      extract_classes=True,
                      extract_img_path=True,
                      extract_shape=True,
                      class_column="Keywords",
                      img_name_col="File",
                      image_path_function=lambda x: x,
                      class_format_function=lambda x: x,
                      train_dir_path="./train",
                      valid_dir_path="./val",
                      test_dir_path="./test",
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
    @default lambda x : x
    @type class_format_function: (class: any) -> 0 | 1
    @param class_format_function: A function that takes object from the class column and that returns 0 (not people) or 1 (people)
    @default default_class_format
    @type train_dir_path: string
    @param train_dir_path: The path of the train directory
    @default "./train"
    @type valid_dir_path: string
    @param valid_dir_path: The path of the validation directory
    @default "./validation"
    @type test_dir_path: string
    @param test_dir_path: The path of the test directory
    @default "./test"
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
    @return: The same as before with the class ("Class"; 0 or 1), dataset type ("Dataset Type"; "train", "validation" or "test") and image path ("Image Path": string) columns added
    """

    if extract_classes:
        print("Pre-processing - Dataset Formatter: beginning class extraction")
        df_with_classes = format_class_names(
            image_df, class_column, class_format_function, final_class_col_name)
        print("Pre-processing - Dataset Formatter: class extraction ended")
    else:
        df_with_classes = image_df
    if extract_img_path:
        print("Pre-processing - Dataset Formatter: beginning path extraction")
        df_with_paths = extract_train_valid_test(df_with_classes, img_name_col, image_path_function,
                                                 train_dir_path, valid_dir_path, test_dir_path, final_type_col_name, final_path_col_name)
        print("Pre-processing - Dataset Formatter: path extraction ended")
    else:
        df_with_paths = df_with_classes
    if extract_shape:
        print("Pre-processing - Dataset Formatter: beginning shape extraction")
        df_with_shapes = extract_height_width(
            df_with_paths, final_path_col_name, final_height_col_name, final_width_col_name)
        print("Pre-processing - Dataset Formatter: shape extraction ended")
    else:
        df_with_shapes = df_with_paths
    return df_with_shapes
