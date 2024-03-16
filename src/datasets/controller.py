from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ControllerDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame):
        super().__init__()

        self._df = df
        self._target_columns = ['steering_tire_angle',
                                'steering_tire_rotation_rate',
                                'acceleration',
                                'speed',
                                'jerk']
        self._feature_columns = list(set(df.columns).difference(set(self._target_columns)))

    def __len__(self):
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._df[self._feature_columns].iloc[row].to_numpy())
        y = torch.from_numpy(self._df[self._target_columns].iloc[row].to_numpy())
        x = x.unsqueeze(dim=0)
        return x, y