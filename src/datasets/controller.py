from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ControllerDataset(Dataset):
    def __init__(self,
                 path_to_file: Path):
        super().__init__()

        self._path_to_file = path_to_file
        self._df = pd.read_csv(path_to_file)

    # TODO: Return x,y of dataset
    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        return None, None