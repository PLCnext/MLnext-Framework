from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler
from matplotlib import rcParams

from .data import detemporalize


def setup_plot():
    """Sets the the global style for plots.
    """
    plt.style.use('ggplot')
    rcParams['figure.figsize'] = (16, 5)
    rcParams['lines.markersize'] = 2

    # color blind friendly cycle (https://gist.github.com/thriveth/8560036)
    rcParams['axes.prop_cycle'] = cycler(color=[
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ])


def plot_error(X: np.array,
               y: np.array = None,
               threshold: float = None,
               title: str = None,
               path: str = None):
    """Plots the error given by x. Label determines the color of the
    point. An anomaly threshold can be drawn with threshold.

    Arguments:
        X (np.array): 2D array of error
        label (np.array): label for data points
        threshold (float): Threshold
        title (str): Title of the plot
        save (str): Path to save figure to

    Example:
        # Plots the predictions X in the color of label with a threshold
        >>> plot_error(X=np.array([0.2, 0.9, 0.4]), y=np.array([0, 1, 0]),
                       threshold=0.5, title='Prediction', path='pred.png')
    """
    plt.title(title)

    if y is None:
        y = np.zeros((X.shape[0], 1))
    y = y[:X.shape[0]]

    plt.xlabel('Sample')
    plt.ylabel('Error')

    plt.scatter(range(X.shape[0]), X, c=y, cmap='jet', zorder=2)
    plt.plot(range(X.shape[0]), X, linewidth=0.5, zorder=1)

    if threshold is not None:
        plt.axhline(y=threshold)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)


def plot_history(history: Dict[str, Any],
                 filter: List[str] = ['loss'],
                 path: str = None):
    """
    Plots the loss in regards to the epoch from the model training history.
    Use filter to plot metrics/losses that contain the filter words.
    Adapted from: https://keras.io/visualization/

    Arguments:
        history (Dict[str, Any]): Training history of a model.
        keywords (List[str]): Filters the history by a keyword.
        path (str): Path to save figure to.

    Example:
        # Plots the training history for entries that match filter
        >>> history = model.fit(...)
        >>> plot_history(history.history, filter=['loss'], path='history.png')
    """
    legend = []
    for key in history.keys():
        if any([keyword in key for keyword in filter]):
            plt.plot(history[key])
            legend.append(key)

    plt.title('Model Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


def _check_inputs(
        x: Optional[Union[np.array, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Transforms `x` into a dataframe.

    Args:
        x (Union[np.array, pd.DataFrame]): Input array.

    Returns:
        pd.DataFrame: Returns a pd.DataFrame.

    """
    if x is None:
        return x

    if isinstance(x, pd.DataFrame):
        return x.reset_index(drop=True)

    x = detemporalize(x, verbose=False)
    return pd.DataFrame(x)


def _truncate_length(*dfs: Optional[pd.DataFrame]) -> List[pd.DataFrame]:
    """Truncates the length of `dfs` to match the shortest in the list.

    Returns:
        List[pd.DataFrame]: Returns a list of dataframe with equal length.
    """
    length = min([df.shape[0] for df in dfs if df is not None])

    return [df.iloc[:length, :] if df is not None else None for df in dfs]


def _get_segments(x: pd.DataFrame, y: pd.DataFrame) -> List[int]:
    """Gets all indexes from `x` where transitions from 0-1 and 1-0 in `y`
    occur.

    Args:
        x (pd.DataFrame): Data.
        y (pd.DataFrame): Labels.

    Returns:
        np.array: Returns a list of indices.
    """
    # get transitions of 0-1 and 1-0
    segments = x.index[(y.iloc[:, -1] > y.shift(1).iloc[:, -1]) |
                       (y.iloc[:, -1] < y.shift(1).iloc[:, -1])].to_list()

    # start and end indexes, last index - 1 because
    # we add +1 when iterating to connect segments
    segments = [0, *segments, x.shape[0] - 1]

    return segments


def plot_signals(*,
                 x_pred: Union[np.array, pd.DataFrame],
                 y: Union[np.array, pd.DataFrame],
                 x: Union[np.array, pd.DataFrame] = None,
                 path: str = None):
    """Plots the signal prediction `x_pred` in color of the label `y`.
    Optionally, `x` is the ground truth.

    Args:
        x_pred (Union[np.array, pd.DataFrame]): Prediction.
        y (Union[np.array, pd.DataFrame]): Labels.
        x (Union[np.array, pd.DataFrame], optional): Ground truth.
        path (str, optional): Path to save fig to.

    Example:
        >>> plot_signals(x_pred=np.zeros((10, 2)),
                         y=np.array([0] * 5 + [1] * 5),
                         x=np.ones((10, 2)),
                         path='signals.png')
    """

    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x_pred.shape[:-1], 1,)))

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x_pred, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x_pred)[-1] > 1 else 1
    rows = -(-np.shape(x_pred)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x_pred)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x_pred
            ax.plot(idx_x,
                    x_pred.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C0')

            # draw x
            if x is not None:
                ax.plot(idx_x, x.iloc[s1:(s2 + 1), idx],
                        c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                        alpha=0.5, zorder=10)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_signals_norm(*,
                      x_pred: Union[np.array, pd.DataFrame],
                      y: Union[np.array, pd.DataFrame],
                      x: Union[np.array, pd.DataFrame] = None,
                      norm_mean: np.array = None,
                      norm_std: np.array = None,
                      path: str = None):
    """Plots prediction `x_pred` in color of the labels `y`. `x` are the inputs
    or ground truth. Additionally, with `norm_mean` and `norm_std` a confidence
    interval can be plotted.

    Args:
        x_pred (Union[np.array, pd.DataFrame]): Prediction.
        y (Union[np.array, pd.DataFrame]): Labels.
        x (Union[np.array, pd.DataFrame], optional): Ground truth.
        norm_mean (np.array, optional): Mean of the underlying normal
        distribution.
        norm_std (np.array, optional): Standard deviation of the underlying
        normal distribution.
        path (str, optional): Path to save figure to.

    Example:
        >>> plot_signals_norm(x_pred=np.zeros((10, 2)),
                              y=np.array([0] * 5 + [1] * 5),
                              x=np.ones((10, 2)),
                              norm_mean=np.array(np.ones((10, 2))),
                              norm_std=np.array(np.ones((10, 2)) * 0.2),
                              path='signals.png')
    """
    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x_pred.shape[:-1], 1,)))

    if (norm_mean is not None) & (norm_std is not None):
        mean = detemporalize(norm_mean, verbose=False)
        std = detemporalize(norm_std, verbose=False)

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x_pred, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x_pred)[-1] > 1 else 1
    rows = -(-np.shape(x_pred)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x_pred)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x_pred
            ax.plot(idx_x,
                    x_pred.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C0')

            # draw x
            if x is not None:
                ax.plot(idx_x,
                        x.iloc[s1:(s2 + 1), idx],
                        c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                        alpha=0.5,
                        zorder=10)

            # plot normal mean and std
            if (mean is not None) & (std is not None):
                ax.plot(idx_x, mean[s1:(s2 + 1), idx],
                        color='C5', lw=.8, zorder=3, alpha=0.5)
                ax.fill_between(idx_x,
                                mean[s1:(s2 + 1), idx] - std[s1:(s2 + 1), idx],
                                mean[s1:(s2 + 1), idx] + std[s1:(s2 + 1), idx],
                                alpha=0.5, color='C5', zorder=3)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_signals_binary(*,
                        x_pred: Union[np.array, pd.DataFrame],
                        y: Union[np.array, pd.DataFrame],
                        x: Union[np.array, pd.DataFrame] = None,
                        bern_mean: np.array = None,
                        path: str = None):
    """Plots signal prediction `x_pred` in color of the labels `y`.
    `x` are the inputs or ground truth. Additionally, with `bern_mean` the mean
    of the underlying bernoulli distribution can be plotted.

    Args:
        x_pred (Union[np.array, pd.DataFrame]): Prediction.
        y (Union[np.array, pd.DataFrame]): Labels.
        x (Union[np.array, pd.DataFrame], optional): Ground truth.
        bern_mean (np.array, optional): Mean of the underlying bernoulli
        distribution.
        path (str, optional): Path to save figure to.

    Example:
        >>> plot_signals_binary(x_pred=np.zeros((10, 2)),
                              y=np.array([0] * 5 + [1] * 5),
                              x=np.ones((10, 2)),
                              bern_mean=np.array(np.ones((10, 2))) * 0.5,
                              path='signals.png')
    """

    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x_pred.shape[:-1], 1,)))

    if bern_mean is not None:
        bern_mean = detemporalize(bern_mean, verbose=False)

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x_pred, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x_pred)[-1] > 1 else 1
    rows = -(-np.shape(x_pred)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x_pred)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x_pred
            ax.plot(idx_x,
                    x_pred.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C0')

            # draw x
            if x is not None:
                ax.plot(idx_x, x.iloc[s1:(s2 + 1), idx],
                        c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                        alpha=0.5, zorder=10)

            # plot bern mean
            if bern_mean is not None:
                ax.plot(idx_x, bern_mean[s1:(s2 + 1), idx], color='C5',
                        lw=.8, zorder=3, alpha=0.5)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()
