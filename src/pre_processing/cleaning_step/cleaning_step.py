from .filter_image_based_on_size import filter_image_with_size, filter_image_with_mode


def cleaning_step(image_df, image_height_col_name="Image Height", image_width_col_name="Image Width",
                  filter_with_size=True, filter_with_size_parameters={"sizes_to_filter": [(3008, 2000), (2000, 3008)]},
                  filter_with_mode=True, filter_with_mode_parameters={"portrait_mode_filtered": True}):
    """
    This function's goal is to filter images from dataframe
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type image_height_col_name: string
    @param image_height_col_name: The col name for image height
    @default "Image Height"
    @type image_width_col_name: string
    @param image_width_col_name: The col name for image width
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

    @rtype: Pandas dataframe
    @return: The filtered dataframe
    """

    indexes_to_filter = []
    for index, row in image_df.iterrows():
        height = row[image_height_col_name]
        width = row[image_width_col_name]
        if (filter_with_size and filter_image_with_size(height, width, **filter_with_size_parameters)) or \
                (filter_with_mode and filter_image_with_mode(height, width, **filter_with_mode_parameters)):
            indexes_to_filter.append(index)
    return image_df.drop(indexes_to_filter, axis=0).reset_index(drop=True)
