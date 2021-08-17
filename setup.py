from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mlnext',
      version='1.0.0',
      author='Phoenix Contact Electronics GmbH',
      author_email='digitalfactorynow@phoenixcontact.com',
      description='Machine learning utilities for Tensorflow/Keras.',
      install_requires=requirements,
      packages=find_packages(include='mlnext'),
      python_requires='>=3.8',
      )
