import argparse
import requests
import yaml
from pathlib import Path
from utils.helpers import load_config

def check(args):
    config_file = args.config
    config = load_config(Path(config_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='config.yaml')
    args = parser.parse_args()
    check(args)
