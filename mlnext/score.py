""" Module for model evaluation.
"""
import typing as T
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from deprecate import deprecated
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics._ranking import _binary_clf_curve

from .anomaly import _recall_anomalies
from .anomaly import apply_point_adjust
from .anomaly import find_anomalies
from .anomaly import recall_anomalies
from mlnext.utils import check_ndim
from mlnext.utils import check_size
from mlnext.utils import truncate

__all__ = [
    'l2_norm',
    'norm_log_likelihood',
    'bern_log_likelihood',
    'kl_divergence',
    'get_threshold',
    'apply_threshold',
    'eval_softmax',
    'eval_sigmoid',
    'moving_average',
    'eval_metrics',
    'eval_metrics_all',
    'ConfusionMatrix',
    'PRCurve',
    'pr_curve',
    'auc_point_adjust_metrics',
    'point_adjust_metrics'
]


def l2_norm(
    x: np.ndarray,
    x_hat: np.ndarray,
    *,
    reduce: bool = True
) -> np.ndarray:
    """Calculates the l2-norm (euclidean distance) for x and x_hat.
    If reduce is False, then the l2_norm is calculated feature-wise.

    Arguments:
        x (np.ndarray): ground truth.
        x_hat (np.ndarray): prediction.

    Returns:
        np.ndarray: Returns the l2-norm between x and x_hat.

    Example:
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]))
        np.ndarray([[0.04]])
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]), reduce=False)
        np.ndarray([0.04, 0.0])
    """
    if reduce:
        r = np.sqrt(np.sum((np.array(x) - np.array(x_hat))**2, axis=-1))
        return r.reshape(-1, 1)
    else:
        return np.sqrt((np.array(x) - np.array(x_hat))**2)


def norm_log_likelihood(
        x: np.ndarray,
        mean: np.ndarray,
        log_var: np.ndarray
) -> np.ndarray:
    """Calculates the negative log likelihood that ``x`` was drawn from a
    normal gaussian distribution defined by ``mean`` and ``log_var``.

    .. math::

        f(x|\\mu, \\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp{-\\frac{1}
        {2}(\\frac{x-\\mu}{\\sigma})^2}

        \\text{Log likelihood}:
        log(f(x | \\mu, \\sigma)) = -0.5 (\\log(2\\pi) + (x-\\mu)^2/\\sigma^2 +
        \\log(\\sigma^2))

    Args:
        x (np.ndarray): Sample.
        mean (np.ndarray): Mean of the gaussian normal distribution.
        log_var (np.ndarray): Log variance of the gaussian normal distribution.

    Returns:
        np.ndarray: Returns the negative log likelihood.
    """
    a = np.log(2. * np.pi) * np.ones(np.shape(x)) + log_var
    b = (x - mean)**2 / (np.exp(log_var) + 1e-10)

    return 0.5 * (a + b)


def bern_log_likelihood(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Calculates the log likelihood of x being produced by a bernoulli
    distribution parameterized by ``mean``.

    .. math::

        LL(x|\\text{mean}) = x \\cdot \\log(\\text{mean}) +
        (1 - x) \\cdot \\log(\\text{mean})

    Args:
        x (np.ndarray): Samples.
        mean (np.ndarray): Mean of the bernoulli distribution.

    Returns:
        np.ndarray: Returns the negative log likelihood.
    """

    a = x * np.log(mean + 1e-10)
    b = (1 - x) * np.log(1 - mean + 1e-10)

    return -(a + b)


def kl_divergence(
    mean: np.ndarray,
    log_var: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1.0
) -> np.ndarray:
    """Calculates the kl divergence kld(q||p) between a normal gaussian ``p``
    (prior_mean, prior_std) and a normal distribution ``q`` parameterized
    by ``mean`` and ``log_var``.

    Args:
        mean (np.ndarray): Mean of q.
        log_var (np.ndarray): Log variance of q.
        prior_mean (float): Mean of the prior p. Defaults to 0.0.
        prior_std (float): Standard deviation of the prior p. Defaults to 1.0.

    Returns:
        np.ndarray: Returns the kl divergence between two normal distributions.
    """

    prior_mean = np.ones(mean.shape) * prior_mean  # type:ignore
    prior_std = np.ones(log_var.shape) * prior_std  # type:ignore

    # see https://stats.stackexchange.com/a/7443
    # KL(q || prior)
    # log(o_p / o_q)
    a = np.log(prior_std) - 0.5 * log_var
    # o_q^2 + (mu_q - mu_p)^2
    b = np.exp(log_var) + np.square(prior_mean - mean)
    # 2o_p^2
    c = 2 * np.square(prior_std)

    return a + (b / c) - 0.5


def get_threshold(x: np.ndarray, *, p: float = 100) -> float:
    """Returns the ``perc``-th percentile of x.

    Arguments:
        x  (np.ndarray): Input
        p (float): Percentage (0-100).

    Returns:
        float: Returns the threshold at the ``perc``-th percentile of x.

    Example:
        >>> get_threshold(np.ndarray([0.0, 1.0]), p=99)
        0.99
    """
    return np.percentile(x, p)


def apply_threshold(
    x: np.ndarray,
    *,
    threshold: float = 0.5,
    pos_label: int = 1
) -> np.ndarray:
    """Applies ``threshold t`` to ``x``. Values that are greater than or equal
    than the ``threshold`` are changed to ``pos_label`` and below to
    ``1 - pos_label``.

    Arguments:
        x (np.ndarray): Input array.
        t (float): Threshold. Defaults to 0.5.
        pos_label (int): Label for the class above the threshold. Defaults to
          1. The other labels is ``1 - pos_label``. Should be either 1 or 0.

    Returns:
        np.ndarray: Returns the result of the threshold operation.

    Example:
        >>> apply_threshold(np.array([0.1, 0.4, 0.8, 1.0]), threshold=0.5)
        np.ndarray([0, 0, 0, 1, 1])
    """
    pos_label = int(pos_label)
    return np.where(x >= threshold, pos_label, 1 - pos_label)


def eval_softmax(y: np.ndarray) -> np.ndarray:
    """Turns a multi-class softmax prediction into class labels.

    Arguments:
        y (np.ndarray): Array with softmax probabilites

    Returns:
        np.ndarray: Returns an array of shape (x, 1) with the class labels.

    Example:
        >>> eval_softmax(np.array([[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]]))
        np.ndarray([[1], [1], [0]])
    """
    return np.argmax(y, axis=-1).reshape(-1, 1)


@deprecated(
    None,
    deprecated_in='0.4',
    remove_in='0.6',
    template_mgs='`%(source_name)s` was deprecated in %(deprecated_in)s '
    'and is removed in %(remove_in)s, use `apply_threshold` instead.'
)
def eval_sigmoid(
    y: np.ndarray,
    *,
    invert: bool = False,
    threshold: float = 0.5
) -> np.ndarray:
    """Turns a binary-class sigmoid prediction into 0-1 class labels.

    Args:
        y (np.ndarray): Array with sigmoid probabilities
        invert (bool): Whether to invert the labels. (0->1, 1->0)
        threshold (float): Threshold in [0, 1]. Default: 0.5

    Returns:
        np.ndarray: Returns the binary class labels.

    Example:
        >>> eval_sigmoid(y=np.array([0.1, 0.6, 0.8, 0.2]))
        np.ndarray([[0],[1],[1],[0]])
    """
    return apply_threshold(
        y,
        threshold=threshold,
        pos_label=0 if invert else 1
    ).reshape(-1, 1)


def moving_average(x: np.ndarray, step: int = 10, mode='full') -> np.ndarray:
    """Calculates the moving average for X with stepsize ``step``.

    Args:
        X (np.ndarray): 1-dimensional array.
        step (int, optional): Stepsize. Defaults to 10.
        mode (str, optional): Mode, see np.convolve. Defaults to 'full'.

    Returns:
        np.ndarray: Returns the moving average.

    Example:
        >>> moving_average(np.array([1, 2, 3, 4]), step=2)
        np.ndarray([0.5, 1.5, 2.5, 3.5, 2.])
    """
    return np.convolve(x, np.ones((step,)) / step, mode=mode)


def eval_metrics(y: np.ndarray, y_hat: np.ndarray) -> T.Dict[str, float]:
    """Calculates accuracy, f1, precision, recall and recall_anomalies.

    Arguments:
        y (np.ndarray): Ground truth labels.
        y_hat (np.ndarray): Predictions (0 or 1).

    Returns:
        T.Dict[str, float]: Returns a dict with all scores.

    Example:
        >>> y, y_hat = np.ones((10, 1)), np.ones((10, 1))
        >>> eval_metrics(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
         'anomalies': 1.0}
    """
    scores = {
        'accuracy': metrics.accuracy_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'f1': metrics.f1_score,
        'anomalies': recall_anomalies
    }

    if y.shape != y_hat.shape:
        warnings.warn(f'Shapes unaligned y {y.shape} and y_hat {y_hat.shape}.')

    (y, y_hat), = truncate((y, y_hat))
    results = {}
    try:
        for key in scores:
            results[key] = scores[key](y, y_hat)
    except Exception:
        pass
    finally:
        return results


def eval_metrics_all(
    y: T.List[np.ndarray],
    y_hat: T.List[np.ndarray]
) -> T.Dict[str, float]:
    """Calculates combined accuracy, f1, precision, recall and AUC scores for
    multiple arrays. The arrays are shorted to the minimum length of the
    corresponding partner and stacked on top of each other to calculated the
    combined scores.

    Arguments:
        y (np.ndarray): Ground truth.
        y_hat (np.ndarray): Prediction.

    Returns:
        T.Dict[str, float]: Returns a dict with all scores.

    Example:
        >>> y = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> y_hat = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> eval_metrics_all(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
         'roc_auc': 1.0}
    """
    if len(y) != len(y_hat):
        raise ValueError(
            'y and y_hat must have the same number elements, '
            f'but found y={len(y)} and y_hat={len(y_hat)}.'
        )

    # allow 1d or 2d arrays with the 2nd dimension of 1
    check_ndim(*y, ndim=2, strict=False)
    check_ndim(*y_hat, ndim=2, strict=False)
    check_size(*y, size=1, axis=1, strict=False)
    check_size(*y_hat, size=1, axis=1, strict=False)

    y = list(map(lambda x: x.reshape(-1), y))
    y_hat = list(map(lambda x: x.reshape(-1), y_hat))

    # truncate corresponding arrays to the same length
    y_, y_hat_ = np.hstack(list(truncate(*zip(y, y_hat))))

    return eval_metrics(y_, y_hat_)


@dataclass
class ConfusionMatrix:
    """``ConfusionMatrix`` is a confusion matrix for a binary classification
    problem. See https://en.wikipedia.org/wiki/Confusion_matrix.

    Args:
      TP (int): true positives, the number of samples from the positive class
        that are correctly assigned to the positive class.
      FN (int): false negatives, the number of samples from the positive
        class that are wrongly assigned to the negative class.
      TN (int): true negatives, the number of samples from the negative class
        that are correctly assigned to negative class.
      FP (int): true negatives, the number of samples from the negative class
        that are wrongly assigned to the positive class.
      DA (int): detected anomalies segments by at least one point.
      TA (int): total number of anomaly segments.
    """
    TP: int = 0  # True Positives
    FN: int = 0  # False Negatives
    TN: int = 0  # True Negative
    FP: int = 0  # False Positive
    DA: int = 0  # Detected anomalies
    TA: int = 0  # total number of anomalies

    def __add__(self, cm: 'ConfusionMatrix') -> 'ConfusionMatrix':
        """Overrides the add operator.

        Returns:
            ConfusionMatrix: Returns a new matrix with feature-wise added
            values.
        """
        return ConfusionMatrix(
            TP=self.TP + cm.TP,
            FN=self.FN + cm.FN,
            TN=self.TN + cm.TN,
            FP=self.FP + cm.FP,
            DA=self.DA + cm.DA,
            TA=self.TA + cm.TA
        )

    def __str__(self) -> str:
        """Creates a string representation of the matrix.

        Returns:
            str: Returns a string representation of the confusion matrix.
        """
        rows = [
            '{:<3s} {:^6s} {:^6s}'.format('P\\A', '1', '0'),
            '{:<3s} {:^6.0f} {:^6.0f}'.format('1', self.TP, self.FP),
            '{:<3s} {:^6.0f} {:^6.0f}'.format('0', self.FN, self.TN),
            *[f'{k}: {v:.4f}' for k, v in self.metrics().items()]
        ]

        return '\n'.join(rows)

    @property
    def accuracy(self) -> float:
        """Calculates the accuracy ``(TP + TN) / (TP + TN + FP + FN)``.

        Returns:
            np.ndarray: Returns the accuracy.
        """
        return ((self.TP + self.TN) /
                (self.TP + self.TN + self.FN + self.FP))

    @property
    def precision(self) -> float:
        """Calculates the precision ``TP / (TP + FP)``.

        Returns:
            float: Returns the precision.
        """
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        """Calculates the recall ``TP / (TP + FN)``.

        Returns:
            float: Returns the recall.
        """
        return self.TP / (self.TP + self.FN)

    @property
    def f1(self) -> float:
        """Calculates the F1-Score
        ``2 * (precision * recall) / (precision + recall)``.

        Returns:
            np.ndarray: Returns the F1-score.
        """
        return ((2 * self.precision * self.recall) /
                (self.precision + self.recall))

    @property
    def recall_anomalies(self) -> float:
        """Calculates the percentage of detected anomaly segments.

        Returns:
            float: Returns the percentage of detected segments.
        """
        return self.DA / self.TA

    def metrics(self) -> T.Dict[str, float]:
        """Returns all metrics.

        Returns:
            T.Dict[str, float]: Returns an mapping of all performance metrics.
        """
        return {
            'accuracy': self.accuracy,
            'f1': self.f1,
            'recall': self.recall,
            'precision': self.precision,
            'anomalies': self.recall_anomalies
        }


@dataclass
class PRCurve:
    """Container for the result of ``pr_curve``. Additionally computes the
    F1-score for each threshold. Can be indexed and returns a
    ``ConfusionMatrix`` for the i-th threshold.

    Args:
        tps (np.ndarray): An increasing count of true positives, at index i
          being the number of positive samples assigned a score >=
          thresholds[i].
        fns (np.ndarray): A count of false negatives, at index i being the
          number of positive samples assigned a score < thresholds[i].
        tns (np.ndarray): A count of true negatives, at index i being the
          number of negative samples assigned a score < thresholds[i].
        fps (np.ndarray): A count of false positives, at index i being the
          number of negative samples assigned a score >= thresholds[i].
        das (np.ndarray): A count of detected anomaly segments, at index i
          being the number of detected anomalies for a score >= thresholds[i].
        tas (np.ndarray): Total number of anomaly segments.
        precision (np.ndarray): Precision values such that element i is the
          precision of predictions with score >= thresholds[i].
        recall (np.ndarray): Decreasing recall values such that element i is
          the recall of predictions with score >= thresholds[i].
        thresholds (np.ndarray): Increasing thresholds on the decision function
          used to compute precision and recall.
    """

    tps: np.ndarray  # true positives
    fns: np.ndarray  # false negatives
    tns: np.ndarray  # true negatives
    fps: np.ndarray  # false positives
    das: np.ndarray  # detected anomaly segments
    tas: np.ndarray  # total number of a anomaly segments

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray

    def __str__(self) -> str:
        """Creates a string representation of the curve.

        Returns:
            str: Returns a string respresentation of the pr-curve.
        """

        header = (' {:^6s} ' * 12).format(
            'TH', 'ACC', 'F1', 'PRC', 'RCL', 'ANO',
            'TP', 'FN', 'TN', 'FP', 'DA', 'TA'
        )

        fmt = ' {:^6.4f} ' * 6 + ' {:^6.0f} ' * 6
        rows = [
            fmt.format(*row)
            for row in
            zip(
                self.thresholds, self.accuracy, self.f1, self.precision,
                self.recall, self.recall_anomalies,
                self.tps, self.fns, self.tns, self.fps, self.das, self.tas
            )
        ]
        return '\n'.join([header, *rows, f'AUC: {self.auc:.4f}'])

    def __getitem__(self, i: int) -> ConfusionMatrix:
        """Creates a confusion matrix for threshold at index i.

        Args:
            idx (int): Threshold index.

        Raises:
            IndexError: Raised if the index is invalid.

        Returns:
            ConfusionMatrix: Returns the confusion matrix for threshold at
            index i.
        """

        if i < 0 or i >= len(self.thresholds):
            raise IndexError(f'Index {i} out of range.')

        return ConfusionMatrix(
            TP=self.tps[i],
            FN=self.fns[i],
            TN=self.tns[i],
            FP=self.fps[i],
            DA=self.das[i],
            TA=self.tas[i]
        )

    def __len__(self) -> int:
        """Retruns the number of thresholds that make up the curve.

        Returns:
            int: Returns the number of thresholds.
        """
        return len(self.thresholds)

    def __iter__(self) -> T.Iterator[ConfusionMatrix]:
        """Creates an iterator over the curve.

        Yields:
            T.Iterator[ConfusionMatrix]: Returns an iterator over the pr curve.
            At index i, the iterator returns a ConfusionMatrix for the i-th
            threshold.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def accuracy(self) -> np.ndarray:
        """Calculates the accuracy where at index i, the accuracy is the
        percentage of correctly assigned samples.

        Returns:
            np.ndarray: Returns the accuracy.
        """
        return ((self.tps + self.tns) /
                (self.tps + self.tns + self.fns + self.fps))

    @property
    def auc(self) -> float:
        """Calculates the area-under-curve (auc).

        Returns:
            float: Returns the area-under-curve (auc) for the precision-recall
            curve.
        """
        # insert (0,1) such that the curve starts from 0
        return auc(np.r_[self.recall, 0], np.r_[self.precision, 1])

    @property
    def f1(self) -> np.ndarray:
        """Calculates the F1-score.

        Returns:
            np.ndarray: Returns the F1-score.
        """
        return ((2 * self.precision * self.recall) /
                (self.precision + self.recall))

    @property
    def recall_anomalies(self) -> np.ndarray:
        """Calculates the fraction of detected anomaly segments.

        Returns:
            np.ndarray: Returns the anomaly recall.
        """
        return self.das / self.tas

    def to_tensorboard(self) -> T.Dict[str, T.Any]:
        """Converts the container to keyword arguments for Tensorboard.
        See https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md.

        Returns:
            T.Dict[str, T.Any]: Returns the pr-curve format expected for
            Tensorboard.
        """  # noqa
        return {
            'true_positive_counts': self.tps,
            'false_positive_counts': self.fps,
            'true_negative_counts': self.tns,
            'false_negative_counts': self.fns,
            'precision': self.precision,
            'recall': self.recall,
            'num_thresholds': len(self.thresholds)
        }


@deprecated(
    True,
    args_mapping={'y_true': 'y'},
    deprecated_in='0.4',
    remove_in='0.6',
)
def pr_curve(
    y: np.ndarray,
    y_score: np.ndarray,
    *,
    y_true: T.Optional[np.ndarray] = None,
    pos_label: T.Optional[T.Union[str, int]] = None,
    sample_weight: T.Optional[T.Union[T.List, np.ndarray]] = None
) -> PRCurve:
    """Computes precision-recall pairs for different probability thresholds for
    binary classification tasks.

    Adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html.
    Changed the return value to PRCurve which encapsulates not only the
    recall, precision and thresholds but also the tps, fps, tns and fns. Thus,
    we can obtain all necessary parameters that are required for the logging of
    a pr-curve in tensorboard (https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md).
    Furthermore, we can you use results for further processing.

    Args:
        y (np.ndarray): Positive lables either {-1, 1} or {0, 1}.
          Otherwise, pos_label needs to be given.
        y_score (np.ndarray):  Target scores in range [0, 1].
        pos_label (int, optional):The label of the positive class.
          When pos_label=None, if y is in {-1, 1} or {0, 1},
          pos_label is set to 1, otherwise an error will be raised.
          Defaults to None.
        sample_weight (T.Union[T.List, np.ndarray], optional): Sample weights.
          Defaults to None.

    Returns:
        PRCurve: Returns a PRCurve container for the results.

    Example:
        >>> import numpy as np
        >>> from mlnext.score import pr_curve
        >>> y = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> curve = pr_curve(y, y_scores)
        >>> print(curve)
          TH     ACC      F1     PRC     RCL     ANO      TP      FN      TN      FP      DA      TA
        0.3500  0.7500  0.8000  0.6667  1.0000  1.0000    2       0       1       1       1       1
        0.4000  0.5000  0.5000  0.5000  0.5000  1.0000    1       1       1       1       1       1
        0.8000  0.7500  0.6667  1.0000  0.5000  1.0000    1       1       2       0       1       1
        AUC: 0.7917

        >>> # access fields
        >>> print(curve.f1, curve.thresholds)
        [0.8        0.5        0.66666667] [0.35 0.4  0.8 ]

        >>> # confusion matrix for a specific threshold
        >>> print(curve[np.argmax(f1)])
        P\\A   1      0
        1     2      1
        0     0      1
        accuracy: 0.7500
        f1: 0.8000
        recall: 1.0000
        precision: 0.6667

        >>> # convert to format for the tensorboard writer
        >>> import tensorflow as tf
        >>> import tensorboard.summary.v1 as tb_summary
        >>> pr_curve_summary = tb_summary.pr_curve_raw_data_op(
        ...    "pr", **curve.to_tensorboard())
        >>> writer = tf.summary.create_file_writer("./tmp/pr_curves")
        >>> with writer.as_default():
        >>>    tf.summary.experimental.write_raw_pb(pr_curve_summary, step=1)

    Result:
        .. image:: ../images/pr_curve.png
           :scale: 50 %
    """  # noqa

    fps, tps, thresholds = _binary_clf_curve(
        y, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    fns = tps[-1] - tps
    tns = fps[-1] - fps

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    das, tas = _recall_anomalies_curve(
        y,
        y_score,
        thresholds=thresholds,
        pos_label=pos_label
    )

    return PRCurve(
        tps[sl], fns[sl], tns[sl], fps[sl], das[sl], tas[sl],
        precision[sl], recall[sl], thresholds[sl]
    )


def point_adjust_metrics(
    *,
    y_hat: np.ndarray,
    y: np.ndarray
) -> pd.DataFrame:
    """Calculates the performance metrics for various ``k`` in [0, 100].

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground truth.

    Returns:
        pd.DataFrame: Returns a dataframe with the k as index and the
        corresponding metrics for each k.

    See Also:
        :meth:``mlnext.plot.plot_point_adjust_metrics``: For plotting the
        results.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> point_adjust_metrics(
        ...   np.array([0, 1, 1, 0]), np.array([0, 1, 1, 1]))
             accuracy  precision    recall   f1   roc_auc
        0        1.00        1.0  1.000000  1.0  1.000000
        1        1.00        1.0  1.000000  1.0  1.000000
        2        1.00        1.0  1.000000  1.0  1.000000
        3        1.00        1.0  1.000000  1.0  1.000000
        4        1.00        1.0  1.000000  1.0  1.000000
        5        1.00        1.0  1.000000  1.0  1.000000
        10       1.00        1.0  1.000000  1.0  1.000000
        15       1.00        1.0  1.000000  1.0  1.000000
        20       1.00        1.0  1.000000  1.0  1.000000
        25       1.00        1.0  1.000000  1.0  1.000000
        30       1.00        1.0  1.000000  1.0  1.000000
        35       1.00        1.0  1.000000  1.0  1.000000
        40       1.00        1.0  1.000000  1.0  1.000000
        45       1.00        1.0  1.000000  1.0  1.000000
        50       1.00        1.0  1.000000  1.0  1.000000
        55       1.00        1.0  1.000000  1.0  1.000000
        60       1.00        1.0  1.000000  1.0  1.000000
        65       1.00        1.0  1.000000  1.0  1.000000
        70       0.75        1.0  0.666667  0.8  0.833333
        75       0.75        1.0  0.666667  0.8  0.833333
        80       0.75        1.0  0.666667  0.8  0.833333
        85       0.75        1.0  0.666667  0.8  0.833333
        90       0.75        1.0  0.666667  0.8  0.833333
        95       0.75        1.0  0.666667  0.8  0.833333
        100      0.75        1.0  0.666667  0.8  0.833333
    """
    # k from 0..100
    k = [*range(0, 5), *range(5, 101, 5)]

    # adjust for each k
    y_hat_adj = [apply_point_adjust(y_hat=y_hat, y=y, k=_k) for _k in k]
    # calculate performance metrics for each k
    metrics = [eval_metrics(y=y, y_hat=_y_hat) for _y_hat in y_hat_adj]

    return pd.DataFrame(metrics, index=k)


def auc_point_adjust_metrics(
    *,
    y_hat: np.ndarray,
    y: np.ndarray
) -> T.Dict[str, float]:
    """Calculates the area under the curve for performance metrics with
    point-adjusted predictions for values of ``k`` in [0,100].

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground truth labels.

    Returns:
        T.Dict[str, float]: Returns a mapping from performance metric to auc.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> auc_point_adjust(
        ...   y_hat=np.array([0, 1, 1, 0]), y=np.array([0, 1, 1, 1]))
        {'auc_accuracy': 0.91875,
         'auc_precision': 1.0,
         'auc_recall': 0.8916666666666666,
         'auc_f1': 0.935,
         'auc_roc_auc': 0.9458333333333333}
    """
    df = point_adjust_metrics(y_hat=y_hat, y=y)
    return {
        f'auc_{column}': auc(df.index, df[column]) / 100
        for column in df
    }


def _recall_anomalies_curve(
    y: np.ndarray,
    y_score: np.ndarray,
    *,
    thresholds: T.List[float],
    pos_label: T.Optional[T.Union[str, int]] = None,
    k: float = 0
) -> T.Tuple[np.ndarray, np.ndarray]:
    # determine positive and negative labels
    pos_label = 1 if pos_label is None else pos_label
    y = y == pos_label

    anomalies = find_anomalies(y)

    # placeholder
    detected = np.zeros(len(thresholds))
    total = np.ones(len(thresholds)) * len(anomalies)

    # count segments where at least k% are detected
    for i, threshold in enumerate(thresholds):
        y_hat = apply_threshold(y_score, threshold=threshold)
        detected[i] = _recall_anomalies(anomalies, y_hat, k=k)

    return detected, total
