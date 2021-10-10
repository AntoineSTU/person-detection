import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


def preprocess_images(image_paths, img_path_col="Image Path", final_size=(600, 700), preprocess_function="InceptionV3", return_loaded_images=False):
    """
    This function's goal is to preprocess and load given images (/!\ don't give too much images)
    @type image_paths: string[] | string | Pandas dataframe | [Numpy array]
    @param image_paths: The image path(s) OR the already loaded images
    @type img_path_col: None | string
    @param img_path_col: The column name containing the image paths if image_paths is a dataframe
    @default "Image Path"
    @type final_size: (int, int)
    @param final_size: The size our training images should have
    @default (600, 700)
    @type preprocess_function: string | (image: Numpy array) -> Numpy array
    @param preprocess_function: The last preprocess function (e.g. put the colors between -1 and 1). String for already declared functions
    @default "InceptionV3"
    @type return_loaded_images: boolean
    @param return_loaded_images: If we should also return the unprocessed images
    @default False

    @rtype: numpy array
    @return: All the loaded and preprocessed images
    """

    # Images may already been loaded
    if isinstance(image_paths, list) and len(image_paths) > 0 and isinstance(image_paths[0], np.ndarray):
        loaded_images = image_paths
    else:
        loaded_images = []
        # We must transform image_paths into a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif isinstance(image_paths, pd.DataFrame):
            image_paths = image_paths[img_path_col].tolist()
        # Then we extract the images
        for path in image_paths:
            img = image.load_img(path, target_size=final_size, )
            loaded_images.append(image.img_to_array(img))
    # Finally we apply the preprocess_function
    if preprocess_function == "InceptionV3":
        preprocessed_images = preprocess_input(np.array(loaded_images))
    else:
        preprocessed_images = preprocess_function(np.array(loaded_images))
    if return_loaded_images:
        return {"loaded_images": loaded_images, "preprocessed_images": preprocessed_images}
    else:
        return preprocessed_images
