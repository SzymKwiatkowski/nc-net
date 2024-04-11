from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class NgramDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 context: int,
                 num_of_values_per_word: int):
        super().__init__()
        self.Context = context
        self._df = df
        self.NumOfValuesPerWord = num_of_values_per_word

    def __len__(self):
        print("test")
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        print("test2")
        x = torch.from_numpy(self._df[0:self.NumOfValuesPerWord*self.Context].iloc[row].to_numpy())
        y = torch.from_numpy(self._df[:-self.NumOfValuesPerWord].iloc[row].to_numpy())
        print(x)
        return x,y