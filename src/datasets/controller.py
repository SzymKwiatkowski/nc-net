from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.pandas_helpers import PandasHelpers
from utils.resource_manager import ResourceManager


class ControllerDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 points_count: int = 271,
                 extraction_points_count: int = 20):
        super().__init__()

        self._df = df
        self._target_columns = ResourceManager.get_targets_column_names()
        self._pos_columns = ResourceManager.get_position_column_names()
        self._extraction_points_count = extraction_points_count
        self._points_count = points_count

        columns_selected, columns_to_drop = PandasHelpers.select_columns_with_patter(
            self._df, ResourceManager.get_regex_point_patterns())
        point_pos_cols, _ = PandasHelpers.select_columns_with_patter(
            self._df, ResourceManager.get_regex_point_position_patterns())
        self._point_pos_cols = point_pos_cols
        self._points_training_cols = columns_selected
        self._feature_columns = self._pos_columns + columns_selected
        all_columns = self._feature_columns + self._target_columns

        columns_to_drop = list(set(df.columns).difference(set(all_columns)))
        self._df.drop(inplace=True, columns=columns_to_drop, axis=1)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        point_poses = np.array(self._df.iloc[row][self._point_pos_cols].to_numpy())
        point_cols_count = len(point_poses) // self._points_count
        position = np.array([self._df.iloc[row][self._pos_columns].to_numpy()])
        point_poses = point_poses.reshape((len(point_poses) // point_cols_count, point_cols_count))

        idx = self.get_closest_points_idx(position, point_poses)
        extracted_points = self.extract_points_data(row, idx)

        # TODO: fill with current velocity instead of zero
        x = np.concatenate([np.array([0]), position.T.flatten(), extracted_points])

        x = torch.from_numpy(x)
        y = torch.from_numpy(self._df[self._target_columns].iloc[row].to_numpy())
        x = x.unsqueeze(dim=0)
        y = y.unsqueeze(dim=0)

        return x.float(), y.float()

    def extract_points_data(self, row: int, start_idx: int) -> np.ndarray:
        """
        :arg row: row idx for dataframe
        :arg start_idx: starting index for extraction
        :rtype: np.ndarray - numpy array of extracted points
        """
        points = np.array(self._df.iloc[row][self._points_training_cols].to_numpy())
        point_cols_count = len(points) // self._points_count
        point_poses = points.reshape((len(points) // point_cols_count, point_cols_count))

        points_extr = np.array([point_poses[(start_idx+i) % self._points_count]
                                for i in range(self._extraction_points_count)])

        return points_extr.flatten()

    @staticmethod
    def get_closest_points_idx(pos: np.ndarray, points_poses: np.ndarray) -> int:
        """
        :arg pos: pose data of 1x7 shape
        :arg points_poses: points with poses of nx7 shape
        :rtype: int - start index
        """
        distances = np.sum((points_poses - pos) ** 2, axis=1)
        idx = np.argmin(distances)
        return idx
