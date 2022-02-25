import codecs
import os
import re

from setuptools import find_packages
from setuptools import setup

###############################################################################

NAME = 'mlnext_framework'
PACKAGES = find_packages(where='.')
HERE = os.path.abspath(os.path.dirname(__file__))
META_PATH = os.path.join('mlnext', '__init__.py')
KEYWORDS = ['mlnext', 'machine', 'learning', 'utilities']
PROJECT_URLS = {
    'Source Code': 'https://github.com/PLCnext/MLnext-Framework',
    'Documentation': 'https://mlnext-framework.readthedocs.io/en/latest/'
}
CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries'
]

with open(os.path.join(HERE, 'requirements.txt')) as f:
    INSTALL_REQUIRES = f.read().splitlines()

EXTRAS_REQUIRE = {
    'tests': [
        'testfixtures',
        'pytest',
        'coverage',
        'scipy'
    ],
}

with open(os.path.join(HERE, 'docs/requirements.txt')) as f:
    EXTRAS_REQUIRE['docs'] = f.read().splitlines()

EXTRAS_REQUIRE['dev'] = (
    EXTRAS_REQUIRE['tests'] + EXTRAS_REQUIRE['docs'] + ['pre-commit']
)

###############################################################################


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), 'rb', 'utf-8') as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))

###############################################################################


VERSION = find_meta('version')

LONG = (
    '=======================================\n'
    '``MLnext``: Machine learning utilities.\n'
    '=======================================\n'
    + read('README.rst')
    + '\n\n'
    + 'Release Information\n'
    + '===================\n\n'
    +
    (re.sub(r'`.*`__', '', changes.group(0))
     if (changes :=
         re.search(
             r'(\d+.\d.\d \(.*?\)\r?\n.*?)\r?\n\r?\n\r?\n----\r?\n\r?\n\r?\n',
             read('CHANGELOG.rst'),
             re.S,
         )) is not None
     else 'No Information')
    + '\n\n'
    + find_meta('copyright')
)


setup(
    name=NAME,
    description=find_meta('description'),
    project_urls=PROJECT_URLS,
    version=VERSION,
    author=find_meta('author'),
    author_email=find_meta('email'),
    maintainer=find_meta('author'),
    maintainer_email=find_meta('email'),
    keywords=KEYWORDS,
    long_description=LONG,
    long_description_content_type='text/x-rst',
    packages=PACKAGES,
    python_requires='>=3.8',
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=EXTRAS_REQUIRE['tests']
)
