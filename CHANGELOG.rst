Changelog
=========

Changes for the upcoming release can be found in the `"changelog" directory <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/tree/main/changelog>`_ in our repository.

..
   Do *NOT* add changelog entries here!

   This changelog is managed by towncrier and is compiled at release time.

   See https://www.attrs.org/en/latest/contributing.html#changelog for details.

.. towncrier release notes start


0.6.0 (2024-11-21)
------------------


Changes
^^^^^^^

- Added missing examples in ``mlnext.score`` and updated other examples.
  `#54 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/54>`__
- Dropped support for python 3.8 and added support for 3.13.
  `#65 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/65>`__


Features
^^^^^^^^

- Added `FeatureCreator`.
  `#63 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/63>`__
- Added ``LengthTransformer`` to pad or truncate inputs to a specific length.
  `#64 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/64>`__


Deprecations
^^^^^^^^^^^^

- Removed argument ``y_true`` from ``mlnext.score.pr_curve``. Use ``y`` instead.
  `#41 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/41>`__
- Removed ``mlnext.score.eval_sigmoid``. Use ``mlnext.score.apply_threshold`` instead.
  `#42 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/42>`__


----


0.5.1 (2024-11-21)
------------------


Changes
^^^^^^^

- Added missing examples in ``mlnext.score`` and updated other examples.
  `#54 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/54>`__
- Dropped support for python 3.8 and added support for 3.13.
  `#65 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/65>`__


Features
^^^^^^^^

- Added `FeatureCreator`.
  `#63 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/63>`__
- Added ``LengthTransformer`` to pad or truncate inputs to a specific length.
  `#64 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/64>`__


Deprecations
^^^^^^^^^^^^

- Removed argument ``y_true`` from ``mlnext.score.pr_curve``. Use ``y`` instead.
  `#41 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/41>`__
- Removed ``mlnext.score.eval_sigmoid``. Use ``mlnext.score.apply_threshold`` instead.
  `#42 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/42>`__


----0.5.1 (2024-06-25)
------------------




Changes
^^^^^^^

- Remove ``print`` statement from ``check_shape``.
  `#61 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/61>`__
- Fix package meta information for pypi release.
  `#62 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/62>`__


----

0.5.0 (2024-06-20)
------------------




Changes
^^^^^^^

- Parameter ``value`` is now optional in ``mlnext.pipeline.Fill`` to allow both filling methods.
  `#46 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/46>`__
- Fixed documentation of ``mlnext.pipeline``.
  `#47 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/47>`__
- Allow ``ColumnSelector`` to be fitted on data if ``keys`` is None.
  `#50 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/50>`__
- Fixed type of parameter ``mapper`` in ``ColumnRename``.
  `#51 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/51>`__
- Changed build system to hatchling and updated dev internals.
  `#58 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/58>`__


Features
^^^^^^^^

- Added ``mlnext.utils.convert_sequences`` which converts sequences inside a mapping to a mapping by the index.
  `#48 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/48>`__
- Added ``mlnext.anomaly.hit_rate_at_p``.
  `#53 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/53>`__
- Add py.typed to indicate mypy/typing support.
  `#56 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/56>`__
- Added parameters ``reduce`` and ``mask`` to ``mlnext.rank_features``.
  `#57 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/57>`__


Deprecations
^^^^^^^^^^^^

- Marked ``mlnext.save_config`` for removal in 0.7.
  `#59 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/59>`__
- Deprecated parameter ``keys`` in favor of ``columns`` in ``ColumnSelector``.
  `#60 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/60>`__

----

0.4.1 (2023-05-12)
------------------




Features
^^^^^^^^

- fixed ``mlnext.find_anomalies`` for scalar inputs (every dimension 1)
  `#52 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/52>`__


----


0.4.0 (2023-01-27)
------------------


Changes
^^^^^^^

- Updated the signature and docstrings of ``mlnext.plot.plot_signals``.
  `#32 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/32>`__
- Refactored ``mlnext.score.eval_sigmoid``.
  `#33 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/33>`__
- Removed ``roc_auc`` from ``mlnext.score.eval_metrics``.
  `#35 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/35>`__
- Fixed inconsistent ``typing`` import.
  `#38 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/38>`__
- Removed ``neg_label`` from ``score.apply_threshold``.
  `#40 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/40>`__


Features
^^^^^^^^

- Added ``mlnext.pipeline.ClippingMinMaxScaler``.
  `#37 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/37>`__
- Added ``mlnext.recall_anomalies`` and integrations for ``mlnext.eval_metrics``, ``mlnext.pr_curve``.
  `#39 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/39>`__


Deprecations
^^^^^^^^^^^^

- Deprecated ``y_true`` in favor of ``y`` in ``mlnext.pr_curve``.
  `#41 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/41>`__
- ``score.apply_threshold`` superceedes the functionality of ``score.eval_sigmoid`` and is removed in 0.6.
  `#42 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/42>`__


----


0.3.1 (2022-02-28)
------------------


Changes
^^^^^^^

- Fixed package name in readme.
  `#31 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/31>`__


----


0.3.0 (2022-02-25)
------------------


Changes
^^^^^^^

- Changed `mlnext.score.apply_threshold` to being inclusive for the positive class.
  `#24 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/24>`__
- Cleaned ``mlnext`` namespace.
  `#28 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/28>`__
- Added parameter ``k`` to ``mlnext.anomaly.apply_point_adjust`` from  https://arxiv.org/abs/2109.05257.
  `#29 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/29>`__
- Remove size check of other dimensions in ``mlnext.utils.truncate``.
  `#30 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/30>`__


Features
^^^^^^^^

- Added ``mlnext.score.pr_curve``.
  `#25 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/25>`__
- Added adaptive ``marker_size`` to ``mlnext.plot_error``.
  `#26 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/26>`__
- Added ``stride`` to ``mlnext.temporalize`` and ``mlnext.detemporalize``.
  `#27 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/27>`__


----


0.2.0 (2021-12-03)
-----------------------


Changes
^^^^^^^

- Removed ``mlnext.io.load_model`` and ``Tensorflow`` dependency.
  `#10 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/10>`__
- Changed the roles of ``x`` and ``x_pred`` in ``plot_signals``.
  `#12 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/12>`__
- Fixed typing of ``np.ndarray``.
  `#20 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/20>`__
- Refactored ``mlnext.io``.
  `#22 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/22>`__


Features
^^^^^^^^

- Added legend to ``plot_signal``.
  `#11 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/11>`__
- Added ``norm_log_likelihood`` and ``bern_log_likelihood``.
  `#13 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/13>`__
- Added ``mlnext.anomaly``.
  `#14 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/14>`__
- Functions in ``mlnext.plot`` now optionally return the figure with ``return_fig``.
  `#15 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/15>`__
- Added ``mlnext.score.kl_divergence`` for two normal distributions.
  `#16 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/16>`__
- Added example images for ``mlnext.plot``.
  `#17 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/17>`__
- Added ``mlnext.anomaly.apply_point_adjust_score``.
  `#18 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/18>`__
- Added MIT license with PLCnext Technology Copyright.
  `#19 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/19>`__
- Added truncation and shape assertion methods in ``mlnext.utils``.
  `#21 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/21>`__
- Added ``mlnext.utils.rename_keys`` and ``mlnext.utils.flatten``.
  `#23 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/23>`__


----


0.1.2 (2021-10-01)
------------------


Features
^^^^^^^^

- Added Digital Factory now introduction and legal notice to documentation.
  `#7 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/7>`__
- Added gradient based feature augmentation.
  `#8 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/8>`__


----

0.1.1 (2021-09-02)
------------------


Changes
^^^^^^^

- Fixed installation of package.
  `#5 <https://gitlab.phoenixcontact.com/vmm-factory-automation/digital-factory/data-collection-storage-evaluation/anomaly-detection/mlnext_framework/-/issues/5>`__


----


0.1.0 (2021-09-02)
------------------

Initial Release.
