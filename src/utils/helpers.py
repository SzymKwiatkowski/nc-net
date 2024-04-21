"""Module providing helpers for simple functionalities not related to model training"""
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    """
    :param path: path to config file
    :rtype: dictionary of parameters specified in yaml file
    """
    with open(path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
