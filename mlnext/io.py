import json
import os
from typing import Any
from typing import Dict

import tensorflow.keras as keras
import yaml
from pydantic import BaseModel


def save_json(*, data: Dict[str, Any], name: str, folder: str = '.'):
    """Saves `data` to a name.json in `folder`.

    Args:
        data (Dict[str, Any]): Data to save.
        folder (str): Path to folder.
        name (str): Name of file.

    Example:
        # Save a dictionary to disk
        >>> save_json(data={'name': 'mlnext'}, name='mlnext.json')
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

    Example:
        # Load a json file
        >>> load_json('mlnext.json')
        {'name': 'mlnext'}
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Path {path} invalid.')

    with open(path, 'r') as file:
        data = json.load(file)

    return data


def save_yaml(*, data: Dict[str, Any], name: str, folder: str = '.'):
    """Saves `data` to a name.yaml in `folder`.

    Args:
        data (Dict[str, Any]): Data to save.
        folder (str): Path to folder.
        name (str): Name of file.

    Example:
        # Save dictionary to yaml
        >>> save_yaml(data={'name': 'mlnext'}, name='mlnext.yaml')
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

    Example:
        # Load a yaml file
        >>> load_yaml('mlnext.yaml')
        {'name': 'mlnext'}
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Path {path} invalid.')

    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def save_config(*, config: BaseModel, name: str, folder: str = '.'):
    """Saves a `pydantic.BaseModel` to `yaml`.

    Args:
        model (BaseModel): Basemodel to save
        folder (str): Path to folder
        name (str): Name of file

    Raises:
        ValueError: Raised if folder is invalid.

    Example:
        # Save a pydantic model to yaml
        >>> class User(pydantic.BaseModel): id: int
        >>> user = User(id=1)
        >>> save_config(config=user)
    """
    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    settings = {
        'exclude_unset': True,
        'exclude_none': True
    }

    data = yaml.safe_load(config.json(**settings))
    save_yaml(data=data, folder=folder, name=name)


def load_model(path: str):
    """Loads a `tf.keras.Model` from `path`.

    Args:
        path (str): Path to model.

    Example:
        # Load keras.Model from disk
        >>> model = load_model('./models/dnn')
    """

    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" not found or not a directory.')

    return keras.models.load_model(path)
