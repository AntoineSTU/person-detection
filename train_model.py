import os
import config
import argparse
import pandas as pd
import logging
import errno
import time
import random
import csv
from datetime import datetime

#######################
### SET UP LOG PATH ###
#######################
# If we want to run the __main__ function, we must setup the log path before loading any other module

# First we must set up the new directory folder for all the reports
if __name__ == "__main__":
    default_run_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    run_id = default_run_id
    report_folder_path = "{}/{}".format(config.REPORTS_FOLDER_PATH, run_id)
    no_folder = 1
    while True:
        try:
            os.mkdir(report_folder_path)
            break
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            time.sleep(random.random())
            no_folder += 1
            run_id = "{}-{}".format(default_run_id, no_folder)
            report_folder_path = "{}/{}".format(
                config.REPORTS_FOLDER_PATH, run_id)
            pass
    print("RUN ID: {}".format(run_id))

    # We can create all the paths for the subfolders and all the corresponding directories
    log_folder_path = "{}/logs".format(report_folder_path)
    os.mkdir(log_folder_path)
    figures_folder_path = "{}/figures".format(report_folder_path)
    os.mkdir(figures_folder_path)
    model_parameters_folder_path = "{}/model_parameters".format(
        report_folder_path)
    os.mkdir(model_parameters_folder_path)

    # Then we must set up the log save path
    LOG_FILE_NAME = "{}/main_logs.log".format(log_folder_path)
    os.environ['LOG_FILE_NAME'] = LOG_FILE_NAME


################
### TRAINING ###
################

from src import create_formatted_dataset, train_model_from_scratch

logging.basicConfig(filename=os.getenv('LOG_FILE_NAME'),
                    format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
    # We add an arg parser
    parser = argparse.ArgumentParser(
        description="Train model for telling if there are people in a picture")
    parser.add_argument("--filter-sizes", type=int, default=1,
                        help="If (3008, 2000) size should be filtered (0/1)")
    parser.add_argument("--filter-portrait", type=int, default=1,
                        help="If portrait mode pictures should be filtered (0/1)")
    parser.add_argument("--nb-images", type=int, default=-1,
                        help="Nb of images to be processed (-1 for all)")
    parser.add_argument("--is-balanced", type=int, default=0,
                        help="If the dataset should be balanced or not (0/1)")
    parser.add_argument("--final-height", type=int, default=600,
                        help="Height of the pictures to be processed (resize)")
    parser.add_argument("--final-width", type=int, default=700,
                        help="Width of the pictures to be processed (resize)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer to be used to train the model")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate of the optimizer")
    parser.add_argument("--loss", type=str, default="binary_crossentropy",
                        help="Loss to be used to train the model")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size on training")
    parser.add_argument("--nb-epochs", type=int, default=1,
                        help="Nb of epochs on training")
    parser.add_argument("--balanced-weights", type=int, default=1,
                        help="If we apply balanced weight on classes when training the model (0/1)")
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="If we add an early stopping algorithm when training the model (based on the accuracy evolution)")
    parser.add_argument("--with-pooling", type=int, default=1,
                        help="If we add a pooling layer before the dense one (0/1)")
    parser.add_argument("--detection-model-save-name", type=str,
                        default="detection_model", help="The name of the saved detection model")
    parser.add_argument("--localization-model-save-name", type=str,
                        default="localization_model", help="The name of the saved localization model")
    args = parser.parse_args()

    if not os.path.isfile(config.PREPROCESSED_CSV_PATH):
        logging.info("Beginning dataset formatting")
        print("Beginning dataset formatting")
        raw_df = pd.read_csv(config.RAW_CSV_PATH)
        pre_processed_df = create_formatted_dataset(raw_df,
                                                    train_dir_path=config.TRAIN_DIR_PATH,
                                                    valid_dir_path=config.VAL_DIR_PATH,
                                                    test_dir_path=config.TEST_DIR_PATH)
        pre_processed_df.to_csv(config.PREPROCESSED_CSV_PATH, index=False)
        print("Dataset formatting ended")
        logging.info("Dataset formatting ended")

    # We must format the file names
    detection_model_name = "{}+{}".format(
        args.detection_model_save_name, run_id)
    detection_model_save_path = "{}/{}".format(
        config.SAVED_MODELS_DIR_PATH, detection_model_name)
    localization_model_name = "{}+{}".format(
        args.localization_model_save_name, run_id)
    localization_model_save_path = "{}/{}".format(
        config.SAVED_MODELS_DIR_PATH, localization_model_name)
    model_parameters_save_path = "{}/params_{}.txt".format(
        model_parameters_folder_path, detection_model_name)

    # Then we can train the model
    train_model_params = {
        "image_df_path": config.PREPROCESSED_CSV_PATH,
        "filter_with_size": args.filter_sizes == 1,
        "filter_with_mode": args.filter_portrait == 1,
        "nb_images": args.nb_images,
        "is_balanced": args.is_balanced == 1,
        "final_size": (args.final_height, args.final_width),
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "batch_size": args.batch_size,
        "num_epo": args.nb_epochs,
        "class_weights": "balanced" if args.balanced_weights == 1 or args.loss == "weighted_binary_crossentropy" else None,
        "early_stopping": args.early_stopping == 1,
        "model_type": "pooling" if args.with_pooling == 1 else "flatten",
        "figures_save_path": figures_folder_path,
        "model_params_save_path": model_parameters_save_path,
        "detection_model_save_path": detection_model_save_path,
        "localization_model_save_path": localization_model_save_path
    }
    result_training = train_model_from_scratch(**train_model_params)
    # Now we can save all of that in the main csv
    csv_args = {"run_id": run_id,
                "filter_with_size": train_model_params["filter_with_size"],
                "filter_with_mode": train_model_params["filter_with_mode"],
                "nb_images": train_model_params["nb_images"],
                "is_balanced": train_model_params["is_balanced"],
                "final_size": train_model_params["final_size"],
                "optimizer": train_model_params["optimizer"],
                "learning_rate": train_model_params["learning_rate"],
                "loss": train_model_params["loss"],
                "batch_size": train_model_params["batch_size"],
                "num_epo": train_model_params["num_epo"],
                "class_weights": train_model_params["class_weights"],
                "early_stopping": train_model_params["early_stopping"],
                "model_type": train_model_params["model_type"],
                "accuracy": result_training["metrics"]["accuracy"],
                "f1_score": result_training["metrics"]["f1_score"]}
    if not os.path.isfile(config.REPORTS_CSV_PATH):
        with open(config.REPORTS_CSV_PATH, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(csv_args.keys())
    with open(config.REPORTS_CSV_PATH, "a") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=csv_args.keys(), delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_args)
    print("End of training")
    logging.info("End of training")
