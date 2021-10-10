import os
import config
import pandas as pd
import argparse

from src import run_trained_model

if __name__ == "__main__":
    # We add an arg parser
    parser = argparse.ArgumentParser(
        description="Generate classes and visualization of people in pictures")
    parser.add_argument("--detection-model-name", type=str, default="detection_model+2021_07_06-16_53_02",
                        help="Name of the detection model to use")
    parser.add_argument("--localization-model-name", type=str, default="flatten",
                        help="Name of the localization model to use. \"pooling\" or \"flatten\" if we should create it here")
    parser.add_argument("--nb-img-with-people", type=int, default=6,
                        help="Nb of images with people to get visualization on")
    parser.add_argument("--nb-img-without-people", type=int, default=0,
                        help="Nb of images without people to get visualization on")
    parser.add_argument("--final-height", type=int, default=600,
                        help="Height of the pictures to be processed (resize)")
    parser.add_argument("--final-width", type=int, default=700,
                        help="Width of the pictures to be processed (resize)")
    parser.add_argument("--threshold-detection", type=str, default="0.5",
                        help="Threshold for the detection model")
    parser.add_argument("--threshold-localization", type=str, default="adaptive",
                        help="Threshold for the localization model")
    parser.add_argument("--image-save-name", type=str, default="images_with_localization.png",
                        help="Name of the images + localization")
    parser.add_argument("--show-picture", type=int, default="1",
                        help="If the picture should be shown (0/1)")
    args = parser.parse_args()

    # First we load the pictures
    image_df = pd.read_csv(config.PREPROCESSED_CSV_PATH)
    df_not_people = image_df[image_df["Class"] == "not_people"]
    df_people = image_df[image_df["Class"] == "people"]
    final_df = pd.concat(
        [df_not_people.iloc[:args.nb_img_without_people], df_people.iloc[:args.nb_img_with_people]])

    # Then we create the model paths
    detection_model = "{}/{}".format(config.SAVED_MODELS_DIR_PATH,
                                     args.detection_model_name)
    if args.localization_model_name == "pooling" or args.localization_model_name == "flatten":
        localization_model = args.localization_model_name
    else:
        localization_model = "{}/{}".format(
            config.SAVED_MODELS_DIR_PATH, args.localization_model_name)

    # And we format the thresholds
    try:
        threshold_detection = float(args.threshold_detection)
    except ValueError:
        threshold_detection = args.threshold_detection
    try:
        threshold_localization = float(args.threshold_localization)
    except ValueError:
        threshold_localization = args.threshold_localization

    # We format the path where to save the picture
    run_id = args.detection_model_name.split("+")[-1]
    report_folder_path = "{}/{}".format(config.REPORTS_FOLDER_PATH, run_id)
    if os.path.isdir(report_folder_path):
        image_save_path = "{}/figures/{}".format(
            report_folder_path, args.image_save_name)
    else:
        image_save_path = "{}/figures/{}_{}".format(
            config.REPORTS_FOLDER_PATH, args.detection_model_name, args.image_save_name)

    # Then we can run show_localization
    run_trained_model(final_df,
                      detection_model,
                      localization_model,
                      real_classes=[
                          0] * args.nb_img_without_people + [1] * args.nb_img_with_people,
                      final_size=(args.final_height, args.final_width),
                      threshold_detection=threshold_detection,
                      threshold_localization=threshold_localization,
                      image_save_path=image_save_path,
                      show_picture=args.show_picture == 1)
