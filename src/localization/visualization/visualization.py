import matplotlib.pyplot as plt
from math import ceil

from .single_image_visualization import get_simple_img_visualization


def generate_localization_images(images, predictions, visualization_function="simple_visualization", image_titles=None, color_coeff=0.5, nb_images_per_row=4, image_save_path="../figures/images_with_localization.png", show_picture=False):
    """
    This function shows all the people localization on images based on predictions.
    @type images: [Numpy array]
    @param images: The images loaded in a numpy array
    @type predictions: Numpy array
    @param predictions: The predictions corresponding to the images
    @type visualization_function: string
    @param visualization_function: The function tu use to show vizualisation on each image
    @default "simple_visualization"
    @type image_titles: None || string list
    @param image_titles: The image titles
    @default None
    @type color_coeff: Float between 0 and 1
    @param color_coeff: How much the areas with people will be colored
    @default 0.5
    @type nb_images_per_row: Integer
    @param nb_images_per_row: How much images per row
    @default 4
    @type image_save_path: string
    @param image_save_path: Where to save the image (nowhere if None)
    @default "../figures/images_with_localization.png"
    @type show_picture: boolean
    @param show_picture: If we should show the images or not
    @default False

    @rtype: None
    @return: Nothing
    """

    if visualization_function == "simple_visualization":
        visualization_function = get_simple_img_visualization
    else:  # By default
        visualization_function = get_simple_img_visualization

    nb_images = len(images)
    _, axes = plt.subplots(nrows=ceil(nb_images/nb_images_per_row), ncols=nb_images_per_row,
                           figsize=(nb_images_per_row*3, ceil(nb_images/nb_images_per_row)*3))
    plt.figure(1)
    for i, pred in enumerate(predictions):
        img = images[i]
        visualization = visualization_function(img, pred, color_coeff)
        axes[i//nb_images_per_row, i % nb_images_per_row].imshow(visualization)
        if image_titles != None:
            axes[i//nb_images_per_row, i %
                 nb_images_per_row].title.set_text(image_titles[i])
        axes[i//nb_images_per_row, i %
             nb_images_per_row].set_xticks([])
        axes[i//nb_images_per_row, i %
             nb_images_per_row].set_yticks([])
    if image_save_path != None:
        plt.savefig(image_save_path)
    if show_picture:
        plt.show()
