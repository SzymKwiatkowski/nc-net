import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetSplits(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_split(df: pd.DataFrame, train_size) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, val_df = train_test_split(df, train_size=train_size, random_state=42)

        test_df, val_df = train_test_split(val_df, train_size=0.5, random_state=42)

        return train_df, val_df, test_df
