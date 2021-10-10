def filter_image_with_size(height, width, sizes_to_filter):
    """
    This function's goal is to check if an image should be filtered based 
    on its size and on sizes to delete
    @type height: int
    @param height: The image height
    @type width: int
    @param width: The image width
    @type sizes_to_filter: (int, int)[]
    @param sizes_to_filter: The size images (height, length) to filter

    @rtype: boolean
    @return: If this image should be filtered or not
    """

    return (height, width) in sizes_to_filter


def filter_image_with_mode(height, width, portrait_mode_filtered):
    """
    This function's goal is to check if an image should be filtered based 
    on its mode (portrait mode or landscape mode)
    @type height: int
    @param height: The image height
    @type width: int
    @param width: The image width
    @type portrait_mode_filtered: boolean
    @param portrait_mode_filtered: If portrait mode should be filtered (else landscape mode will be)

    @rtype: boolean
    @return: If this image should be filtered or not
    """

    return height/width > 1 if portrait_mode_filtered else height/width < 1
