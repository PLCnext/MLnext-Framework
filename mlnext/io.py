""" Module for loading and saving files.
"""
import glob
import json
import os
from typing import Any
from typing import Dict
from typing import List

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
        >>> # Save a dictionary to disk
        >>> save_json(data={'name': 'mlnext'}, name='mlnext.json')
    """

    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    filename, ext = os.path.splitext(name)
    if not ext:
        name = f'{filename}.json'
    elif ext not in {'.json'}:
        raise ValueError(f'Invalid extension "{ext}".')

    with open(os.path.join(folder, name), mode='w') as file:
        json.dump(data, file, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Loads a `.json` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        Dict[str, Any]: Returns the loaded json.

    Example:
        >>> # Load a json file
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
        >>> # Save dictionary to yaml
        >>> save_yaml(data={'name': 'mlnext'}, name='mlnext.yaml')
    """

    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a valid directory.')

    filename, ext = os.path.splitext(name)
    if not ext:
        name = f'{filename}.yaml'
    elif ext not in {'.yaml', '.yml'}:
        raise ValueError(f'Invalid extension "{ext}".')

    with open(os.path.join(folder, name), mode='w') as file:
        yaml.dump(data, file, indent=2, sort_keys=False)


def load_yaml(path: str) -> Dict[str, Any]:
    """Loads a `.yaml`/`.yml` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        Dict[str, Any]: Returns the loaded yaml file.

    Example:
        >>> # Load a yaml file
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
        >>> # Save a pydantic model to yaml
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
        >>> # Load keras.Model from disk
        >>> model = load_model('./models/dnn')
    """

    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" not found or not a directory.')

    return keras.models.load_model(path)


def load(path: str) -> Dict[str, Any]:
    """Loads a file from `path` with the supported python parser.

    Args:
        path (str): Path to file.

    Raises:
        ValueError: Raised if

    Returns:
        Dict[str, Any]: Returns the content.

    Example:
        >>> # Loads file from path
        >>> load('./resources/task.json')
        {
            "name": "task",
            ...
        }
    """
    _, ext = os.path.splitext(path)

    exts = {
        '.json': load_json,
        '.yaml': load_yaml,
        '.yml': load_yaml
    }

    if ext not in exts:
        raise ValueError(f'Incompatible extension "{ext}".'
                         f'Supported extensions: {exts.keys()}.')

    return exts[ext](path)


def get_files(*,
              path: str,
              extension: str,
              full_path: bool = False) -> List[str]:
    """List all files in `path` with `extension`.

    Args:
        path (str): Path of directory.
        extension (str): File extension (without dot).
        full_path (bool): Whether to return the full path or only the
        filenames.

    Raises:
        ValueError: Raised if `path` is not a directory.

    Returns:
        List[str]: Returns a list of files of `extension` in `path`.

    Example:
        >>> # lists all files in dir
        >>> get_files(path='./resources/tasks', extension='json')
        ['task.json']

        >>> get_files(path='.resources/tasks', extension='json',
        ... full_path=True)
        ['.../resources/tasks/task.json']
    """
    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" is not a directory.')

    files = glob.glob(f'{path}/*.{extension}')

    if full_path:
        return files

    return list(map(os.path.basename, files))


def get_folders(path: str, *, full_path: bool = False) -> List[str]:
    """Lists all folders in a directory.

    Args:
        path (str): Path of directory.
        full_path (bool): Whether to return the full path or only the folder
        names.

    Raises:
        ValueError: Raised if `path` is not a directory.

    Returns:
        List[str]: Returns a list of the names of the folders.

    Example:
        >>> # list all folder in a directory
        >>> get_folders('./resources')
        ['tasks', 'models']

        >>> get_folders('./resources', full_path=True)
        ['.../resources/tasks', '.../resources/models']
    """
    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" is not a directory.')

    return [name if not full_path else os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]
