import json
import os
from typing import Any
from typing import Dict

import yaml
from pydantic import BaseModel


def save_json(*, data: Dict[str, Any], folder: str, name: str):
    """Saves `data` to a name.json in `folder`.

    Args:
        data (Dict[str, Any]): Data to save.
        folder (str): Path to folder.
        name (str): Name of file.
    """

    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    if '.json' not in name:
        name += '.json'

    with open(os.path.join(folder, name), mode='w') as file:
        json.dump(data, file, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Loads a `.json` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        Dict[str, Any]: Returns the loaded json.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Path {path} invalid.')

    with open(path, 'r') as file:
        data = json.load(file)

    return data


def save_yaml(*, data: Dict[str, Any], folder: str, name: str):
    """Saves `data` to a name.yaml in `folder`.

    Args:
        data (Dict[str, Any]): Data to save.
        folder (str): Path to folder.
        name (str): Name of file.
    """

    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    if '.yaml' not in name:
        name += '.yaml'

    with open(os.path.join(folder, name), mode='w') as file:
        yaml.dump(data, file, indent=2, sort_keys=False)


def load_yaml(path: str) -> Dict[str, Any]:
    """Loads a `.yaml`/`.yml` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        Dict[str, Any]: Returns the loaded yaml file.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Path {path} invalid.')

    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def save_config(*, config: BaseModel, folder: str, name: str):
    """Saves a `pydantic.BaseModel` to `yaml`.

    Args:
        model (BaseModel): Basemodel to save
        folder (str): Path to folder
        name (str): Name of file

    Raises:
        ValueError: Raised if folder is invalid.
    """
    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    settings = {
        'exclude_unset': True,
        'exclude_none': True
    }

    data = yaml.safe_load(config.json(**settings))
    save_yaml(data=data, folder=folder, name=name)
