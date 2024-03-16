import yaml
from pathlib import Path


def load_config(path: Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config
