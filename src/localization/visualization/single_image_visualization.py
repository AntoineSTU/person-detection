import numpy as np


def get_simple_img_visualization(img, prediction, color_coeff):
    """
    This function shows all the people localization on one image, by coloring where the people are.
    @type img: Numpy array
    @param img: The images loaded in a numpy array
    @type prediction: Numpy array
    @param prediction: The prediction corresponding to the image
    @type color_coeff: Float between 0 and 1
    @param color_coeff: How much the areas with people will be colored 

    @rtype: Numpy array
    @return: Image with localization
    """

    img_array = img.copy()
    height_img, width_img, _ = img_array.shape
    height_pred, width_pred = prediction.shape
    i_step = height_img/height_pred
    j_step = width_img/width_pred
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i, j] == 1:
                i_0 = int(i*i_step)
                i_1 = int((i+1)*i_step)
                j_0 = int(j*j_step)
                j_1 = int((j+1)*j_step)
                green_array = np.array(
                    [[[0, 255, 0] for k in range(j_1-j_0)] for l in range(i_1-i_0)])
                img_array[i_0:i_1, j_0:j_1] = (
                    1-color_coeff) * img_array[i_0:i_1, j_0:j_1] + color_coeff * green_array
    return img_array.astype('uint8')
