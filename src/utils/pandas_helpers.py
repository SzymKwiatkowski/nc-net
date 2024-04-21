"""Module providing helpers to processing pandas dataframes."""

import re
import pandas as pd


class PandasHelpers:
    """Class providing helpers to processing pandas dataframes."""
    @staticmethod
    def select_columns(
            df: pd.DataFrame,
            points_pattern: list[str],
            points_count: int) -> tuple[list[str], list[str]]:
        """
        :arg df: Pandas dataframe
        :arg points_pattern: list of pattern strings eg. point_{index}_orientation_z
        :arg points_count: number of points to select in the dataframe
        :rtype: tuple[list[str], list[str]]
        """
        cols = list(df.columns)

        generate_regex_matches = [pattern.format(index=str(i))
                                  for i in range(points_count)
                                  for pattern in points_pattern]
        print(generate_regex_matches)
        match_cols = []
        drop_cols = []

        for col in cols:
            if any(re.match(col, match) for match in generate_regex_matches):
                match_cols.append(col)
            else:
                drop_cols.append(col)

        return match_cols, drop_cols

    @staticmethod
    def select_columns_with_patter(
            df: pd.DataFrame,
            patterns: list[str]) -> tuple[list[str], list[str]]:
        """
        :arg df: Pandas dataframe
        :arg patterns: list of regex pattern strings
        :rtype: tuple[list[str], list[str]]
        """
        cols = list(df.columns)
        match_cols = []
        drop_cols = []
        for col in cols:
            matches = [re.search(match, col) for match in patterns]
            if any(matches):
                match_cols.append(col)
            else:
                drop_cols.append(col)

        return match_cols, drop_cols
