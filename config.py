import os
from datetime import datetime

IS_LOCAL = True

if IS_LOCAL:
    RAW_CSV_PATH = "./data/raw/test_preprocess/RAISE_6k_extract.csv"
    PREPROCESSED_CSV_PATH = "./data/interim/test_preprocess/df_formatted_main.csv"
    DEFAULT_LOG_FOLDER_PATH = "./reports/logs"
    REPORTS_FOLDER_PATH = "./reports"
    SAVED_MODELS_DIR_PATH = "./models"
    TRAIN_DIR_PATH = "./data/raw/test_preprocess/Train"
    VAL_DIR_PATH = "./data/raw/test_preprocess/Valid"
    TEST_DIR_PATH = "./data/raw/test_preprocess/Test"
    REPORTS_CSV_PATH = "./reports/train_results.csv"
else:
    RAW_CSV_PATH = "/original/RAISE/RAISE_all.csv"
    PREPROCESSED_CSV_PATH = "/scratch/projekt1/data/interim/df_formatted_main.csv"
    DEFAULT_LOG_FOLDER_PATH = "/scratch/projekt1/reports/logs"
    REPORTS_FOLDER_PATH = "/scratch/projekt1/reports"
    SAVED_MODELS_DIR_PATH = "/scratch/projekt1/models"
    TRAIN_DIR_PATH = "/original/RAISE/RaiseTrain"
    VAL_DIR_PATH = "/original/RAISE/RaiseVal"
    TEST_DIR_PATH = "/original/RAISE/RaiseTest"
    REPORTS_CSV_PATH = "/scratch/projekt1/reports/train_results.csv"

# We must setup the log file name for all the files (by default)
LOG_FILE_NAME = "{}/{}.log".format(DEFAULT_LOG_FOLDER_PATH,
                                   datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
os.environ['LOG_FILE_NAME'] = LOG_FILE_NAME
