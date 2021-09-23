
MLNext Framework
================

*MLNext Framework* is an open source framework for hardware independent execution of
machine learning using *Python* and *Docker*.
It provides machine learning utilities for Tensorflow and Keras.
The corresponding *Python* package is called *mlnext*.
MLNext Framework belongs to a solution portfolio for the Digital Factory now to realize Data collection, storage, and evaluation.

Digitalization is posing numerous challenges for production – but above all, it provides countless opportunities for increasing productivity and system availability. To ensure that you benefit from the advantages of digitalization as quickly as possible, we will provide you with needs-based support – from installing simple stand-alone solutions to comprehensive digitalization concepts.

The Digital Factory now is based on the following four fields of activity:

- Data collection, storage, and evaluation
- Data transportation
- Data security
- Data usage

The four fields of action provide you with various solutions, from data acquisition to data utilization. Each individual solution will not only be tailored to your particular requirements; the fields of action can also be combined in any way or considered individually. Regardless of which path you are taking toward the Digital Factory, we will be happy to support you during the next steps.

To help you to meet today’s digitalization challenges and implement opportunities profitably, our solutions provide the following added values:

- Scalability – tailored to your requirements
- Tested and validated – in our own in-house production facilities
- Ready-to-use – benefit from the digital transformation today

With target-oriented consultation, we will find the right solution for your Digital Factory together. Let us take on the challenges of digitalization and leverage its opportunities together.


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
