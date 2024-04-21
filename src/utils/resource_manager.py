
class ResourceManager:
    @staticmethod
    def get_regex_point_patterns() -> list[str]:
        return ["point_(\d*)_pos\w*", "point_(\d*)_orientation\w*"]

    @staticmethod
    def get_regex_point_position_patterns() -> list[str]:
        return ["point_(\d*)_pos\w*", "point_(\d*)_orientation_\w*"]

    @staticmethod
    def get_targets_column_names() -> list[str]:
        return ['steering_tire_angle',
                'steering_tire_rotation_rate',
                'acceleration',
                'speed',
                'jerk']

    @staticmethod
    def get_position_column_names() -> list[str]:
        return ["pose_x", "pose_y", "pose_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"]
