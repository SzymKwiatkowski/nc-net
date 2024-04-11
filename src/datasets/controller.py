from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from src.utils.pandas_helpers import PandasHelpers
from src.utils.resource_manager import ResourceManager


class ControllerDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 points_count: int = 20):
        super().__init__()

        self._df = df
        self._target_columns = ResourceManager.get_targets_column_names()
        self._pos_columns = ResourceManager.get_position_column_names()

        columns_selected, columns_to_drop = PandasHelpers.select_columns_with_patter(
            self._df, ResourceManager.get_regex_point_patterns())
        self._feature_columns = self._pos_columns + columns_selected
        all_columns = self._feature_columns + self._target_columns
        columns_to_drop = list(set(df.columns).difference(set(all_columns)))
        self._df.drop(inplace=True, columns=columns_to_drop, axis=1)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._df[self._feature_columns].iloc[row].to_numpy())
        y = torch.from_numpy(self._df[self._target_columns].iloc[row].to_numpy())
        x = x.unsqueeze(dim=0)
        return x, y
