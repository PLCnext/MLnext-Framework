""" Module for loading and saving files.
"""
import glob
import json
import os
import typing as T

import yaml
from pydantic import BaseModel

__all__ = [
    'save_json',
    'load_json',
    'save_yaml',
    'load_yaml',
    'save_config',
    'load',
    'get_files',
    'get_folders'
]


def save_json(data: T.Dict[str, T.Any], *, name: str, folder: str = '.'):
    """Saves `data` to a name.json in `folder`.

    Args:
        data (T.Dict[str, T.Any]): Data to save.
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


def load_json(path: str) -> T.Dict[str, T.Any]:
    """Loads a `.json` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        T.Dict[str, T.Any]: Returns the loaded json.

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


def save_yaml(data: T.Dict[str, T.Any], *, name: str, folder: str = '.'):
    """Saves `data` to a name.yaml in `folder`.

    Args:
        data (T.Dict[str, T.Any]): Data to save.
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


def load_yaml(path: str) -> T.Dict[str, T.Any]:
    """Loads a `.yaml`/`.yml` file from `path`.

    Args:
        path (str): Path to file.

    Returns:
        T.Dict[str, T.Any]: Returns the loaded yaml file.

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


def save_config(config: BaseModel, *, name: str, folder: str = '.'):
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

    data = yaml.safe_load(config.json(**settings))  # type: ignore
    save_yaml(data=data, folder=folder, name=name)


def load(path: str) -> T.Dict[str, T.Any]:
    """Loads a file from `path` with the supported python parser.

    Args:
        path (str): Path to file.

    Raises:
        ValueError: Raised if

    Returns:
        T.Dict[str, T.Any]: Returns the content.

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


def get_files(
    path: str,
    *,
    name: str = '*',
    ext: str = '*',
    absolute: bool = False
) -> T.List[str]:
    """T.List all files in `path` with extension `ext`.

    Args:
        path (str): Path of the directory.
        ext (str): File extension (without dot).
        name (str): Pattern for the name of the files to appear in the result.
        absolute (bool): Whether to return the absolute path or only the
          filenames.

    Raises:
        ValueError: Raised if `path` is not a directory.

    Returns:
        T.List[str]: Returns a list of files with extension `ext` in `path`.

    Example:
        >>> # lists all files in dir
        >>> get_files(path='./resources/tasks', ext='json')
        ['task.json']

        >>> # get all files named task
        >>> get_files(path='./resources/tasks', name='task')
        ['task.json', 'task.yaml']

        >>> # get the absolute path of the files
        >>> get_files(path='.resources/tasks', ext='json',
        ... absolute=True)
        ['.../resources/tasks/task.json']
    """
    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" is not a directory.')

    files = glob.glob(f'{path}/{name}.{ext}')

    if absolute:
        return files

    return list(map(os.path.basename, files))  # type: ignore


def get_folders(
    path: str,
    *,
    filter: str = '',
    absolute: bool = False
) -> T.List[str]:
    """Lists all folders in `folder`.

    Args:
        path (str): Path of the directory.
        filter (str): Pattern to match the beginning of the folders names.
        absolute (bool): Whether to return the absolute path or only the
          foldernames.

    Raises:
        ValueError: Raised if `folder` is not a directory.

    Returns:
        T.List[str]: Returns a list of the names of the folders.

    Example:
        >>> # list all folder in a directory
        >>> get_folders('./resources')
        ['tasks', 'models']

        >>> # Get all folders that start with the letter m
        >>> get_folders('./resources', filter='m')
        ['models']

        # Get the absolute path of the folders
        >>> get_folders('./resources', absolute=True)
        ['.../resources/tasks', '.../resources/models']

    """
    if not os.path.isdir(path):
        raise ValueError(f'Path "{path}" is not a directory.')

    return [name if not absolute else os.path.join(path, name)
            for name in os.listdir(path)
            if (os.path.isdir(os.path.join(path, name))
                and name.startswith(filter))]
