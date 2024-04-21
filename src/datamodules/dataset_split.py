"""Class for splitting the dataset into train/val and test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split


@staticmethod
def basic_split(df: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param df: dataframe containing data to be split.
    :param train_size: train size for split.
    :return: dataframe containing train, val, and test data.
    """

    train_df, val_df = train_test_split(df, train_size=train_size, random_state=42)
    test_df, val_df = train_test_split(val_df, train_size=0.5, random_state=42)

    return train_df, val_df, test_df
