|MLnext| MLnext Framework
=========================

.. |MLnext| image:: https://raw.githubusercontent.com/PLCnext/MLnext-Framework/main/docs/source/_static/mlnext.png

|pypi| |python| |mypy| |license| |Digital Factory Now| |docs|


.. |pypi| image:: https://badge.fury.io/py/mlnext-framework.svg
   :target: https://pypi.org/project/mlnext-framework/
   :alt: Latest Version

.. |python| image:: https://img.shields.io/pypi/pyversions/mlnext-framework
    :target: https://www.python.org/
    :alt: Supported Python Versions

.. |mypy| image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: MyPy checked

.. |license| image:: https://img.shields.io/github/license/PLCnext/MLnext-Framework.svg
   :target: https://github.com/PLCnext/MLnext-Framework/blob/master/LICENSE
   :alt: License

.. |Digital Factory Now| image:: https://img.shields.io/badge/Digital%20Factory-Now-blue.svg
   :target: https://www.phoenixcontact.com/online/portal/pi?1dmy&urile=wcm:path:/pien/web/offcontext/insite_landing_pages/176a6497-e932-4acc-87bc-798c7a9f8aad/176a6497-e932-4acc-87bc-798c7a9f8aad

.. |docs| image:: https://readthedocs.org/projects/mlnext-framework/badge/?version=latest
   :target: https://mlnext-framework.readthedocs.io/en/latest/?badge=latest

*MLnext Framework* is an open source framework for hardware independent execution of machine learning using *Python* and *Docker*.
It provides machine learning utilities.
The corresponding *Python* package is called *mlnext-framework*.
MLnext Framework belongs to a solution portfolio for the Digital Factory now to realize Data collection, storage, and evaluation.

Digitalization is posing numerous challenges for production --- but above all, it provides countless opportunities for increasing productivity and system availability.
To ensure that you benefit from the advantages of digitalization as quickly as possible, we will provide you with needs-based support --- from installing simple stand-alone solutions to comprehensive digitalization concepts.

The Digital Factory now is based on the following four fields of activity:

- Data collection, storage, and evaluation
- Data transportation
- Data security
- Data usage

The four fields of action provide you with various solutions, from data acquisition to data utilization. Each individual solution will not only be tailored to your particular requirements; the fields of action can also be combined in any way or considered individually.
Regardless of which path you are taking toward the Digital Factory, we will be happy to support you during the next steps.

To help you to meet today's digitalization challenges and implement opportunities profitably, our solutions provide the following added values:

- Scalability --- tailored to your requirements
- Tested and validated --- in our own in-house production facilities
- Ready-to-use --- benefit from the digital transformation today

With target-oriented consultation, we will find the right solution for your Digital Factory together. Let us take on the challenges of digitalization and leverage its opportunities together.


Installation
------------

Install this package using ``pip``\ :

.. code-block:: bash

   pip install mlnext-framework

Modules
-------

The *MLnext Framework* consists of 7 modules:

.. code-block:: python

   import mlnext.data as data          # for data loading and manipulation
   import mlnext.io as io              # for loading and saving files
   import mlnext.pipeline as pipeline  # for data preprocessing
   import mlnext.plot as plot          # for data visualization
   import mlnext.score as score        # for model evaluation
   import mlnext.anomaly as anomaly    # for anomaly interpretation
   import mlnext.utils as utils        # for utility functions

   # hint: all functions can also be accessed from the root module
   import mlnext


Development
-----------

MLnext uses `rye <https://rye.astral.sh/>`_ to manage the development environment.
Install rye by following the instructions on their website and run ``rye sync`` to setup the development environment.

Furthermore, we use black and ruff to enforce style standards on the codebase.
The formatting is done for you via pre-commit, and is enforced via the ``tox -e lint`` in the CI/CD.
Run ``pre-commit install`` to set up the git hooks;
subsequently, when you git commit, the formatter will be run. If the
changed files are not conformant, the hook will have reformatted them and
you may need to run pre-commit again. You can run ``pre-commit run --all-files``
to manually run the formatters.

Build the documentation by running ``tox -e docs``.
