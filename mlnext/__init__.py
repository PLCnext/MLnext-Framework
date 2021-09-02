"""Machine learning utilities for Tensorflow/Keras.
"""
from datetime import date

from .data import *  # noqa
from .io import *  # noqa
from .pipeline import *  # noqa
from .plot import *  # noqa
from .score import *  # noqa

__version__ = '0.2.0.dev0'

__title__ = 'MLnext'
__description__ = 'Machine learning utilities for Tensorflow/Keras.'

__author__ = 'Phoenix Contact Electronics GmbH'
__email__ = 'digitalfactorynow@phoenixcontact.com'

__copyright__ = f'Copyright (c) {date.today().year} {__author__}'
