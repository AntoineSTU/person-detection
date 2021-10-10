import pandas as pd
from PIL import Image


def extract_height_width(image_df, path_col_name, final_height_col_name, final_width_col_name):
    """
    This function's goal is to add in the dataframe the heights and widths of the images.
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type path_col_name: string
    @param path_col_name: The column name containing the image paths
    @param final_height_col_name: string
    @type final_height_col_name: The name of the final image height column
    @param final_width_col_name: string
    @type final_width_col_name: The name of the final image width column

    @rtype: Pandas dataframe
    @return: The same as before with the image height ("Image Height": int) and image width ("Image Width": int) columns added
    """

    heights = []
    widths = []
    nb_of_images = len(image_df.index)
    for index, row in image_df.iterrows():
        picture_path = row[path_col_name]
        img = Image.open(picture_path)
        (width, height) = img.size
        heights.append(height)
        widths.append(width)
        img.close()
        if (index+1) % 1000 == 0:
            print(
                "Pre-processing - Shape extractor: Extracted {} images out of {}".format(index+1, nb_of_images))
    modified_df = image_df.copy(deep=False)
    modified_df[final_height_col_name] = pd.DataFrame(
        heights, index=modified_df.index)
    modified_df[final_width_col_name] = pd.DataFrame(
        widths, index=modified_df.index)
    return modified_df
