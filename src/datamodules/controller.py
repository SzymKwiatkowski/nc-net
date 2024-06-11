"""Datamodule for managing dataset data and workers during training"""
from pathlib import Path
from typing import Optional
import json
import pandas as pd
from lightning import pytorch as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from datamodules.dataset_split import basic_split
from datasets.controller import ControllerDataset
from datasets.dataset_config import DatasetConfig
from utils.resource_manager import ResourceManager


# pylint: disable=R0913, R0902
class ControllerDataModule(pl.LightningDataModule):
    """Class for managing dataset data and workers during training"""
    def __init__(
            self,
            data_path: Path,
            batch_size: int = 32,
            extraction_points_count: int = 10,
            num_workers: int = 4,
            model_type: str = None):
        super().__init__()

        self._data_path = Path(data_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._extraction_points_count = extraction_points_count
        self._target_columns = ResourceManager.get_targets_column_names()
        self._pos_columns = ResourceManager.get_position_column_names_short()
        self._point_poses_columns = ResourceManager.get_regex_point_position_patterns_short()
        self.model_type = model_type
        self.n_targets = len(self._target_columns)
        self.n_features = len(self._point_poses_columns) * (extraction_points_count + 1)

        self.train_dataset, self.test_dataset, self.val_dataset = self.prepare_datasets(
            self.get_configs(self._data_path / "datasets.json"))

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

    def prepare_datasets(self, dataset_configs: Optional[list[DatasetConfig]]) -> tuple[Dataset, Dataset, Dataset]:
        """
        Prepares dataset for each config
        :param dataset_configs: list of dataset configs
        """
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for dataset_config in dataset_configs:
            df = pd.read_csv(self._data_path / dataset_config.main_df)
            race_track_df = pd.read_csv(self._data_path / dataset_config.points_df)
            train_df, val_df, test_df = basic_split(df, dataset_config.train_size)
            train_datasets.append(ControllerDataset(
                train_df,
                race_track_df,
                pos_columns=self._pos_columns,
                target_columns=self._target_columns,
                point_poses_columns=self._point_poses_columns,
                points_count=dataset_config.points_count,
                extraction_points_count=self._extraction_points_count,
                model_type=self.model_type,
            ))

            val_datasets.append(ControllerDataset(
                val_df,
                race_track_df,
                pos_columns=self._pos_columns,
                target_columns=self._target_columns,
                point_poses_columns=self._point_poses_columns,
                points_count=dataset_config.points_count,
                extraction_points_count=self._extraction_points_count,
                model_type=self.model_type,
            ))

            test_datasets.append(ControllerDataset(
                test_df,
                race_track_df,
                pos_columns=self._pos_columns,
                target_columns=self._target_columns,
                point_poses_columns=self._point_poses_columns,
                points_count=dataset_config.points_count,
                extraction_points_count=self._extraction_points_count,
                model_type=self.model_type,
            ))

        return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)

    @staticmethod
    def get_configs(config_path: Path) -> list[DatasetConfig]:
        """
        Returns configs loaded from config file
        :param config_path: path to config file
        """
        with open(config_path, encoding='utf-8') as configs:
            configs = json.load(configs)
        return [DatasetConfig(config) for config in configs]
