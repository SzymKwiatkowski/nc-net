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
        self._feature_columns = list(set(self._target_columns).difference(set(df.columns)))

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._df[self._feature_columns], self._df[self._target_columns]