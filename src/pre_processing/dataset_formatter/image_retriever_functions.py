import os
import pandas as pd


def extract_train_valid_test(image_df, img_name_col, image_path_function, train_dir_path, valid_dir_path, test_dir_path, final_type_col_name, final_path_col_name):
    """
    This function's goal is to add in the dataframe if if from training, validation or 
    testing sets and to add the image path in the dataframe.
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type img_name_col: string
    @param img_name_col: The column name containing the image names
    @type image_path_function: (image_name: any, directory_path: string) -> string
    @param image_path_function: A function returning image paths based on their names (cf. name col) and the directory path
    @type train_dir_path: string
    @param train_dir_path: The path of the train directory
    @type valid_dir_path: string
    @param valid_dir_path: The path of the validation directory
    @type test_dir_path: string
    @param test_dir_path: The path of the test directory
    @param final_type_col_name: string
    @type final_type_col_name: The name of the final dataset type column
    @param final_path_col_name: string
    @type final_path_col_name: The name of the final image path column

    @rtype: Pandas dataframe
    @return: The same as before with the dataset type ("Dataset Type"; "train", "validation" or "test") and image path ("Image Path": string) columns added
    """

    dataset_type_col = []
    path_col = []
    for _, row in image_df.iterrows():
        im_path_train = image_path_function(row[img_name_col], train_dir_path)
        if os.path.isfile(im_path_train):
            dataset_type_col.append("train")
            path_col.append(im_path_train)
        else:
            im_path_valid = image_path_function(
                row[img_name_col], valid_dir_path)
            if os.path.isfile(im_path_valid):
                dataset_type_col.append("validation")
                path_col.append(im_path_valid)
            else:
                im_path_test = image_path_function(
                    row[img_name_col], test_dir_path)
                if os.path.isfile(im_path_test):
                    dataset_type_col.append("test")
                    path_col.append(im_path_test)
                else:
                    dataset_type_col.append(None)
                    path_col.append(None)
    modified_df = image_df.copy(deep=False)
    modified_df[final_type_col_name] = pd.DataFrame(
        dataset_type_col, index=modified_df.index)
    modified_df[final_path_col_name] = pd.DataFrame(
        path_col, index=modified_df.index)
    return modified_df
