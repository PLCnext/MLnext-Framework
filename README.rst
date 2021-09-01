
MLNext Framework
================

*MLNext Framework* is an open source framework for hardware independent execution of
machine learning using *Python* and *Docker*.
It provides machine learning utilities for Tensorflow and Keras.
The corresponding *Python* package is called *mlnext*.

Installation
------------

Install this package using ``pip``\ :

.. code-block:: bash

   pip install mlnext --index-url https://pypi:ZS2HLWUqbgmjfURn6U_7@gitlab.phoenixcontact.com/api/v4/projects/771/packages/pypi/simple --trusted-host gitlab.phoenixcontact.com

Modules
-------

The *MLnext Framework* consists of 5 modules:

.. code-block:: python

   import mlnext.data as data          # for data loading and manipulation
   import mlnext.io as io              # for loading and saving files
   import mlnext.pipeline as pipeline  # for data preprocessing
   import mlnext.plot as plot          # for data visualization
   import mlnext.score as score        # for model evaluation
