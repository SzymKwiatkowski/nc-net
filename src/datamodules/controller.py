"""Datamodule for managing dataset data and workers during training"""
from pathlib import Path
from typing import Optional

import pandas as pd
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from datamodules.dataset_split import basic_split
from datasets.controller import ControllerDataset


# pylint: disable=R0913, R0902
class ControllerDataModule(pl.LightningDataModule):
    """Class for managing dataset data and workers during training"""
    def __init__(
            self,
            data_path: Path,
            batch_size: int = 32,
            num_workers: int = 4,
            train_size: float = 0.8,
            points_count: int = 271,
            extraction_points_count: int = 20):
        super().__init__()

        self._data_path = Path(data_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_size = train_size
        self._points_count = points_count
        self._extraction_points_count = extraction_points_count

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        df = pd.read_csv(self._data_path / "main_df.csv")
        race_track_df = pd.read_csv(self._data_path / "points_df.csv")
        train_df, val_df, test_df = basic_split(df, self._train_size)

        self.train_dataset = ControllerDataset(
            train_df,
            race_track_df,
            self._points_count,
            self._extraction_points_count
        )

        self.val_dataset = ControllerDataset(
            val_df,
            race_track_df,
            self._points_count,
            self._extraction_points_count
        )

        self.test_dataset = ControllerDataset(
            test_df,
            race_track_df,
            self._points_count,
            self._extraction_points_count
        )

        self.n_features = self.train_dataset.n_features
        self.n_targets = self.train_dataset.n_targets
        self.save_hyperparameters(ignore=['data_path', 'number_of_workers'])

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )
