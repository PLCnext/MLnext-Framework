from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mlutils',
      version='1.0',
      author='Gorden Platz',
      author_email='gplatz@phoenixcontact.com',
      description='Machine learning utilities for Tensorflow/Keras.',
      install_requires=requirements,
      packages=find_packages(include='mlutils'),
      python_requires='>=3.8',
      )
