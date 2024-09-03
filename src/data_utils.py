import os
from typing import Tuple, List, Dict

import gdown
import pandas as pd

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download from GDrive all the needed datasets for the project.

    Returns:
        train : pd.DataFrame
            Training dataset

        test : pd.DataFrame
            Test dataset
    """

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    train = pd.read_csv(config.DATASET_TRAIN)
    test = pd.read_csv(config.DATASET_TEST)

    return train, test


def split_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Args:
        train : pd.DataFrame
            Training dataset.
        test : pd.DataFrame
            Test dataset.

    Returns:
        X_train : pd.Series
            List reviews for train

        y_train : pd.Series
            List labels for train

        X_test : pd.Series
            List reviews for test

        y_test : pd.Series
            List labels for test
    """
    return (train["review"], train["positive"], test["review"], test["positive"])

def get_class_percentages(df: pd.DataFrame, class_name: str) -> Dict[str, float]:
    """This function returns the class percentages in the target column
    Args: 
        df : pd.DataFrame the dataset to analyze
        class_name: str the name of the target column

    Returns:
        List[float, ...] list with the parcentage of each class 
    """

    len_of_df = df.shape[0]
    dict_class_percent = {}
    #df.loc[:, class_name].value_count().lambda(percent, count: dict_class_percent[percent]=count*100/len_of_df)
    return dict_class_percent
    
