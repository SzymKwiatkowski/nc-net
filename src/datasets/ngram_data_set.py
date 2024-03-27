from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class NgramDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 Context: int,
                 NumOfValuesPerWord: int):
        super().__init__()
        self.Context=Context
        self._df = df
        #print(len(self._df))
        self.NumOfValuesPerWord=NumOfValuesPerWord
    def __len__(self):
        #print("test")
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        #print(self._df[0:self.NumOfValuesPerWord*self.Context])
        x = torch.from_numpy(self._df[0:self.NumOfValuesPerWord*self.Context].iloc[row].to_numpy())
        y = torch.from_numpy(self._df[-self.NumOfValuesPerWord:].iloc[row].to_numpy())
        print(x)
        return x,y