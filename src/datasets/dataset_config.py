"""Class for specifing dataset configuration"""


class DatasetConfig:
    """Class mapping json properties into class properties"""
    def __init__(self, json_obj):
        """Initialize class with specified properties"""
        self.main_df = json_obj['main_df']
        self.points_df = json_obj['points_df']
        self.points_count = json_obj['points_count']
        self.train_size = json_obj['train_size']

