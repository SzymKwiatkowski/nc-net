"""Module for preparing data for training and testing."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from utils.resource_manager import ResourceManager
from utils.pandas_helpers import PandasHelpers


def valid_df(df: pd.DataFrame, selected_cols) -> bool:
    """Validates if every saved row has the same values"""
    numpy_array = df[selected_cols].to_numpy()
    diff = np.sum(numpy_array - numpy_array[0])
    if diff > 0:
        return False

    return True


def prepare_dataset(args) -> None:
    """Prepares dataset for training and testing."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(data_dir)

    pose_cols = ResourceManager.get_position_column_names()
    target_cols = ResourceManager.get_targets_column_names()
    points_selected_cols, _ = PandasHelpers.select_columns_with_patter(
            df, ResourceManager.get_regex_point_position_patterns_for_all_cols())

    main_df_cols = pose_cols + target_cols
    main_df = df[main_df_cols].copy()
    points_df = df[points_selected_cols].copy()

    if not valid_df(points_df, points_selected_cols):
        print("Dataset is not valid")
        return

    main_df.to_csv(output_dir / 'main_df.csv', index=False)
    race_path = pd.DataFrame(points_df.loc[[0], :])
    race_path.to_csv(output_dir / 'points_df.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='prepare_dataset',
        description='Prepare dataset for training and testing.',
        epilog='')
    # Points to root project dir
    parser.add_argument('-d', '--data_dir', action='store', default='../data/sample_data.csv')
    # Using neptune
    parser.add_argument('-o', '--output-dir', action='store', default=False)

    args_parsed = parser.parse_args()
    prepare_dataset(args_parsed)
