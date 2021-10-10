def format_class_names(image_df, class_column, format_function, final_class_col_name):
    """
    This function's goal is to format the binary classes in the dataframe 
    (0 for not_people, 1 for people)
    @type image_df: Pandas dataframe
    @param image_df: The dataframe containing the images
    @type class_column: string
    @param class_column: The column name from which we can extract the classes
    @type format_function: (class: any) -> "people"|"not_people"
    @param format_function: A function that takes object from the class column and that returns 0 (not people) or 1 (people)
    @param final_class_col_name: string
    @type final_class_col_name: The name of the final class column

    @rtype: Pandas dataframe
    @return: The same df with a new column "Class" with the results of the format function
    """

    modified_df = image_df.copy(deep=False)
    modified_df[final_class_col_name] = image_df[class_column].map(
        format_function)
    return modified_df
