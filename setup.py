from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mlutils',
      version='1.0',
      install_requires=requirements,
      packages=find_packages(include='mlutils')
      )
