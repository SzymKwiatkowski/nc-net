"""Implementation of dataset for controller network."""
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.pandas_helpers import PandasHelpers


# pylint: disable=R0902, R0913
class ControllerDataset(Dataset):
    """Class implementing dataset for controller network."""
    def __init__(self,
                 df: pd.DataFrame,
                 points_df: pd.DataFrame,
                 target_columns: list['str'],
                 pos_columns: list['str'],
                 point_poses_columns: list['str'],
                 points_count: int = 271,
                 extraction_points_count: int = 10,
                 model_type: str = None):
        super().__init__()

        self._df = df
        self._points_df = points_df
        self._target_columns = target_columns
        self._pos_columns = pos_columns
        self._point_poses_columns = point_poses_columns
        self._extraction_points_count = extraction_points_count
        self._points_count = points_count
        self._model_type = model_type
        self._max_y = 0.523599

        columns_selected, _ = PandasHelpers.select_columns_with_patter(
            self._points_df, self._point_poses_columns)
        self._point_pos_cols = columns_selected
        self._points_training_cols = columns_selected
        self._feature_columns = self._pos_columns + columns_selected

        point_cols_count = len(self._point_pos_cols) // self._points_count
        self.n_features = self._extraction_points_count * point_cols_count + len(self._pos_columns)
        self.n_targets = len(self._target_columns)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        point_poses = np.array(self._points_df[self._point_pos_cols].to_numpy()).T
        point_cols_count = len(point_poses) // self._points_count
        position = np.array([self._df.iloc[row][self._pos_columns].to_numpy()])
        point_poses = point_poses.reshape((len(point_poses) // point_cols_count, point_cols_count))

        idx = self.get_closest_points_idx(position, point_poses)
        extracted_points = self.extract_points_data(idx)

        x = np.concatenate([position.T.flatten(), extracted_points])

        x = torch.from_numpy(x)
        y = (torch.from_numpy((self._df[self._target_columns].iloc[row].to_numpy() + self._max_y) / (2 * self._max_y)))

        return x.float(), y.float()

    def extract_points_data(self, start_idx: int) -> np.ndarray:
        """
        :arg row: row idx for dataframe
        :arg start_idx: starting index for extraction
        :rtype: np.ndarray - numpy array of extracted points
        """
        points = np.array(self._points_df[self._points_training_cols].to_numpy()).T
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
