"""Module providing resources to other scripts"""


class ResourceManager:
    """Class providing resources to other scripts"""
    @staticmethod
    def get_regex_point_position_patterns() -> list[str]:
        """
        :return: list of regex point patterns for point extraction of
        """
        return [
            r"point_(\d*)_pos\w*",
            r"point_(\d*)_orientation_\w*"
        ]

    @staticmethod
    def get_targets_column_names() -> list[str]:
        """
        :return: list of target column names
        """
        return [
            r'steering_tire_angle',
            r'steering_tire_rotation_rate',
            r'acceleration',
            r'speed',
            r'jerk'
        ]

    @staticmethod
    def get_position_column_names() -> list[str]:
        """
        :return: list of position columns names
        """
        return [
            r"pose_x",
            r"pose_y",
            r"pose_z",
            r"orientation_x",
            r"orientation_y",
            r"orientation_z",
            r"orientation_w"
        ]
