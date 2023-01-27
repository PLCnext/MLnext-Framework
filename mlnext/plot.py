""" Module for data visualization.
"""
import typing as T
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cycler
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import auc
from sklearn.preprocessing import minmax_scale

from .anomaly import rank_features
from .data import detemporalize
from .score import point_adjust_metrics
from .utils import RangeDict

__all__ = [
    'setup_plot',
    'plot_error',
    'plot_history',
    'plot_signals',
    'plot_signals_norm',
    'plot_signals_binary',
    'plot_rankings',
    'plot_point_adjust_metrics'
]


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


marker_sizes = RangeDict({
    range(0, 2499): 4,
    range(2500, 4999): 3.5,
    range(5000, 7499): 3,
    range(7500, 9999): 2.5
})


def _adaptive_makersize(num_points: int) -> float:
    try:
        return marker_sizes[num_points]
    except KeyError:
        return 2.0


def plot_error(
    X: np.ndarray,
    y: T.Optional[np.ndarray] = None,
    threshold: T.Optional[float] = None,
    title: T.Optional[str] = None,
    yscale: str = 'linear',
    path: T.Optional[str] = None,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """Plots the error given by `x`. Label determines the color of the
    point. An anomaly threshold can be drawn with threshold.

    Arguments:
        X (np.ndarray): 2D array of error.
        label (np.ndarray): label for data points.
        threshold (float): Threshold.
        title (str): Title of the plot.
        yscale (str): Scaling of y-axis; passed to plt.yscale. Default: linear.
        save (str): Path to save figure to.
        return_fig (bool): Whether to return the figure. Otherwise,
          plt.show() is called.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns the figure if return_fig
        is true.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> # Plots the predictions X in the color of label with a threshold
        >>> mlnext.plot_error(
        ...     X=np.random.rand(10),
        ...     y=np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0]),
        ...     threshold=0.5,
        ...     title='Prediction',
        ...     path='pred.png'
        ... )

    Expected result:
        .. image:: ../images/plot_error.png
           :scale: 50 %
    """
    fig = plt.figure()
    plt.title(title)
    plt.yscale(yscale)

    if y is None:
        y = np.zeros((X.shape[0], 1))

    if X.shape[0] != y.shape[0]:
        raise Warning('Length misaligned: '
                      f'X ({X.shape[0]}) and y ({y.shape[0]}).')

    y = y[:X.shape[0]]

    plt.xlabel('Sample')
    plt.ylabel('Error')

    plt.scatter(
        range(X.shape[0]), X,
        c=y, cmap='jet', zorder=2,
        s=_adaptive_makersize(X.shape[0])**2)
    plt.plot(range(X.shape[0]), X, linewidth=0.5, zorder=1)

    if threshold is not None:
        plt.axhline(y=threshold)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_history(
    history: T.Dict[str, T.Any],
    filter: T.List[str] = ['loss'],
    path: T.Optional[str] = None,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """
    Plots the loss in regards to the epoch from the model training history.
    Use filter to plot metrics/losses that contain the filter words.
    Adapted from: https://keras.io/visualization/

    Arguments:
        history (T.Dict[str, T.Any]): Training history of a model.
        keywords (T.List[str]): Filters the history by a keyword.
        path (str): Path to save figure to.
        return_fig (bool): Whether to return the figure. Otherwise,
          plt.show() is called.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns the figure if return_fig
          is true.

    Example:
        >>> import numpy as np
        >>> # Plots the training history for entries that match filter
        >>> history = model.fit(...)
        >>> mlnext.plot_history(
        ...  history.history, filter=['val'], path='history.png')

    Expected result:
        .. image:: ../images/plot_history.png
           :scale: 50 %
    """
    fig = plt.figure()
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

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def _check_inputs(
    x: T.Optional[T.Union[np.ndarray, pd.DataFrame]]
) -> T.Optional[pd.DataFrame]:
    """Transforms `x` into a dataframe.

    Args:
        x (T.Union[np.ndarray, pd.DataFrame], optional): Input array.

    Returns:
        pd.DataFrame: Returns a pd.DataFrame.

    """
    if x is None:
        return x

    if isinstance(x, pd.DataFrame):
        return x.reset_index(drop=True)

    x = detemporalize(x, verbose=False)
    return pd.DataFrame(x)


def _truncate_length(*dfs: T.Optional[pd.DataFrame]) -> T.List[pd.DataFrame]:
    """Truncates the length of `dfs` to match the shortest in the list.

    Returns:
        T.List[pd.DataFrame]: Returns a list of dataframe with equal length.
    """
    length = min([df.shape[0] for df in dfs if df is not None])

    return [df.iloc[:length, :] if df is not None else None for df in dfs]


def _get_segments(x: pd.DataFrame, y: pd.DataFrame) -> T.List[int]:
    """Gets all indexes from `x` where transitions from 0-1 and 1-0 in `y`
    occur.

    Args:
        x (pd.DataFrame): Data.
        y (pd.DataFrame): Labels.

    Returns:
        np.ndarray: Returns a list of indices.
    """
    # get transitions of 0-1 and 1-0
    segments = x.index[(y.iloc[:, -1] > y.shift(1).iloc[:, -1]) |
                       (y.iloc[:, -1] < y.shift(1).iloc[:, -1])].to_list()

    # start and end indexes, last index - 1 because
    # we add +1 when iterating to connect segments
    segments = [0, *segments, x.shape[0] - 1]

    return segments


def plot_signals(
    x: T.Union[np.ndarray, pd.DataFrame],
    y: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    *,
    x_pred: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    path: T.Optional[str] = None,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """Plots the signal `x` in color of the label `y`.
    Optionally, `x_pred` is signal prediction to be plotted.

    Args:
        y (T.Union[np.ndarray, pd.DataFrame]): Labels.
        x (T.Union[np.ndarray, pd.DataFrame]): Ground truth. Default: None.
        x_pred (T.Union[np.ndarray, pd.DataFrame], optional): Prediction.
          Default: None.
        path (str, optional): Path to save fig to. Default: T.Optional.
        return_fig (bool): Whether to return the figure. Otherwise,
          plt.show() is called. Default: False.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns the figure if return_fig
        is true.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> mlnext.plot_signals(
        ...     x_pred=np.zeros((10, 2)),
        ...     y=np.array([0] * 5 + [1] * 5),
        ...     x=np.ones((10, 2)),
        ...     path='signals.png'
        ... )

    Expected result:

        .. image:: ../images/plot_signals.png
           :scale: 50 %
    """

    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x.shape[:-1], 1,)))  # type: ignore

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x)[-1] > 1 else 1
    rows = -(-np.shape(x)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x
            ax.plot(idx_x,
                    x.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                    label='x')

            # draw x_pred
            if x_pred is not None:
                ax.plot(idx_x, x_pred.iloc[s1:(s2 + 1), idx],
                        c='C0',
                        alpha=0.8, zorder=10, label='x_pred')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', ncol=len(by_label))
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_signals_norm(
    x: T.Union[np.ndarray, pd.DataFrame],
    y: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    *,
    x_pred: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    norm_mean: T.Optional[np.ndarray] = None,
    norm_std: T.Optional[np.ndarray] = None,
    path: T.Optional[str] = None,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """Plots the signal `x` in color of the label `y`. Optionally,
    `x_pred` is signal prediction to be plotted. Additionally, with `norm_mean`
    and `norm_std` a confidence interval can be plotted.

    Args:
        x_pred (T.Union[np.ndarray, pd.DataFrame]): Prediction.
        y (T.Union[np.ndarray, pd.DataFrame]): Labels. Default: None.
        x (T.Union[np.ndarray, pd.DataFrame], optional): Ground truth.
          Default: None.
        norm_mean (np.ndarray, optional): Mean of the underlying normal
          distribution.
        norm_std (np.ndarray, optional): Standard deviation of the underlying
          normal distribution.
        path (str, optional): Path to save figure to. Default: None.
        return_fig (bool): Whether to return the figure. Otherwise,
          plt.show() is called. Default: False.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns the figure if return_fig
        is true.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> mlnext.plot_signals_norm(
        ...     x_pred=np.zeros((10, 2)),
        ...     y=np.array([0] * 5 + [1] * 5),
        ...     x=np.ones((10, 2)),
        ...     norm_mean=np.array(np.ones((10, 2))),
        ...     norm_std=np.array(np.ones((10, 2)) * 0.2),
        ...     path='signals.png'
        ... )

    Expected result:

        .. image:: ../images/plot_signals_norm.png
           :scale: 50 %
    """
    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x.shape[:-1], 1,)))  # type: ignore

    if norm_mean is not None and norm_std is not None:
        mean = detemporalize(norm_mean, verbose=False)
        std = detemporalize(norm_std, verbose=False)

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x)[-1] > 1 else 1
    rows = -(-np.shape(x)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x
            ax.plot(idx_x,
                    x.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                    label='x')

            # draw x_pred
            if x_pred is not None:
                ax.plot(idx_x,
                        x_pred.iloc[s1:(s2 + 1), idx],
                        c='C0',
                        label='x_pred')

            # plot normal mean and std
            if (mean is not None) & (std is not None):
                ax.plot(idx_x, mean[s1:(s2 + 1), idx],
                        color='C5', lw=.8, zorder=3, alpha=0.8, label='mean')
                ax.fill_between(idx_x,
                                mean[s1:(s2 + 1), idx] - std[s1:(s2 + 1), idx],
                                mean[s1:(s2 + 1), idx] + std[s1:(s2 + 1), idx],
                                alpha=0.5, color='C5', zorder=3, label='std')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', ncol=len(by_label))

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_signals_binary(
    x: T.Union[np.ndarray, pd.DataFrame],
    y: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    *,
    x_pred: T.Optional[T.Union[np.ndarray, pd.DataFrame]] = None,
    bern_mean: T.Optional[np.ndarray] = None,
    path: T.Optional[str] = None,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """Plots the signal `x` in color of the label `y`. Optionally, `x_pred` is
    signal prediction to be plotted. `x` are the inputs or ground truth.
    Additionally, with `bern_mean` the mean of the underlying bernoulli
    distribution can be plotted.

    Args:
        x_pred (T.Union[np.ndarray, pd.DataFrame]): Prediction.
        y (T.Union[np.ndarray, pd.DataFrame]): Labels. Default: None.
        x (T.Union[np.ndarray, pd.DataFrame], optional): Ground truth.
          Default: None.
        bern_mean (np.ndarray, optional): Mean of the underlying bernoulli
          distribution. Default: None.
        path (str, optional): Path to save figure to. Default: None.
        return_fig (bool): Whether to return the figure. Otherwise,
          plt.show() is called. Default: False.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns the figure if return_fig
        is true.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> mlnext.plot_signals_binary(
        ...    x_pred=np.zeros((10, 2)),
        ...    y=np.array([0] * 5 + [1] * 5),
        ...    x=np.ones((10, 2)),
        ...    bern_mean=np.array(np.ones((10, 2))) * 0.5,
        ...    path='signals.png'
        ... )

    Expected result:

        .. image:: ../images/plot_signals_binary.png
           :scale: 50 %
    """

    x_pred = _check_inputs(x_pred)
    x = _check_inputs(x)
    y = _check_inputs(y)

    if y is None:
        y = pd.DataFrame(np.zeros((*x.shape[:-1], 1,)))  # type: ignore

    if bern_mean is not None:
        bern_mean = detemporalize(bern_mean, verbose=False)

    x_pred, x, y = _truncate_length(x_pred, x, y)
    segments = _get_segments(x, y)

    # plot grid n x 2 if more than plot
    columns = 2 if np.shape(x)[-1] > 1 else 1
    rows = -(-np.shape(x)[-1] // columns)

    # prepare subplots
    figsize = (7.5 * columns, 2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten() if columns > 1 else [axes]
    plt.subplots_adjust(hspace=0.5)

    # iterate over segments
    for s1, s2 in zip(segments[:-1], segments[1:]):
        # draw in color of label for each col
        for idx, (ax, col) in enumerate(zip(axes, x)):
            ax.set_title(col)

            idx_x = np.array(range(int(s1), int(s2 + 1)))

            # draw x
            ax.plot(idx_x,
                    x.loc[s1:s2, col],
                    c='C1' if y.iloc[s1, 0] > 0. else 'C2',
                    label='x')

            # draw x_pred
            if x_pred is not None:
                ax.plot(idx_x,
                        x_pred.iloc[s1:(s2 + 1), idx],
                        c='C0',
                        label='x_pred')

            # plot bern mean
            if bern_mean is not None:
                ax.plot(idx_x, bern_mean[s1:(s2 + 1), idx], color='C5',
                        lw=.8, zorder=3, alpha=0.8, label='bern_mean')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', ncol=len(by_label))

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_rankings(
    error: np.ndarray,
    *,
    y: np.ndarray,
    x: np.ndarray,
    x_pred: np.ndarray,
    k=3,
    context=10,
    return_figs: bool = False
) -> T.Optional[T.List[Figure]]:
    """Plots the top k features (predictions `x_pred` and ground truth `x`)
    with the biggest error for each anomaly found in the labels `y`. With
    context additional data points to the left and right of the anomaly will
    be shown.

    Args:
        error (np.ndarray): Feature-wise error (e.g. l2_norm with reduce=False
          or {norm, bern}_log_likelihood).
        y (np.ndarray): Labels (1d array).
        x (np.ndarray): Input data (ground truth).
        x_pred (np.ndarray): Predictions of x.
        k (int, optional): How many features should be plotted. Defaults to 3.
        context (int, optional): Additional datapoints that should be plotted
          to the left and right of the anomaly. Defaults to 10.
        return_figs (bool): Whether to return the figure. Otherwise, plt.show()
          is called.

    Returns:
        T.Optional[matplotlib.figure.Figure]: Returns a list of the figures if
        return_figs is true.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> x, x_pred = np.ones((7, 4)), np.random.random((7, 4))
        >>> y = np.array([0, 0, 1, 1, 1, 0, 0])
        >>> error = mlnext.l2_norm(x, x_pred, reduce=False)
        >>> mlnext.plot_rankings(
        ...    error = error,
        ...    y=y,
        ...    x=x,
        ...    x_pred=x_pred,
        ...    k=2
        ... )

    Expected result:

        .. image:: ../images/plot_rankings.png
           :scale: 50 %
    """
    error = detemporalize(error, verbose=False)
    x = detemporalize(x, verbose=False)
    x_pred = detemporalize(x_pred, verbose=False)
    y = detemporalize(y, verbose=False).squeeze()

    anm, rks, merr = rank_features(error=error, y=y)

    columns = min(2, k)
    rows = -(-min(k, error.shape[-1]) // columns) + 1

    figs = []
    for (s, e), ranks, mean_error in zip(anm, rks, merr):
        # prepare subplots
        figsize = (7.5 * columns, rows * 2)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        figs.append(fig)
        plt.subplots_adjust(hspace=0.5)

        gs = fig.add_gridspec(nrows=rows, ncols=columns)
        axes = [fig.add_subplot(gs[x, y])
                for (x, y) in product(range(rows - 1), range(columns))]

        fig.suptitle(f'Anomaly: {(s, e)}')
        fig.legend(handles=[Patch(color='C2', label='x'),
                            Patch(color='C0', label='x_pred')],
                   ncol=2, loc='upper center',
                   bbox_to_anchor=(0.5, 0.95))

        s = max(s - context, 0)
        e = min(e + context + 1, x.shape[0])
        idx_x = np.array(range(s, e))

        for ax, r in zip(axes, ranks):
            r = int(r)

            cmap = LSC.from_list('x', [(0, 'C2'), (1, 'C1')])
            ax.add_collection(_create_lc(idx_x, x[s:e, r], y[s:e], cmap))

            cmap = LSC.from_list('x_pred', [(0, 'C0'), (1, 'C0')])
            ax.add_collection(_create_lc(idx_x, x_pred[s:e, r], y[s:e], cmap))

            ax.autoscale()
            ax.set_title(r)

        table_ax = fig.add_subplot(gs[rows - 1, :])
        table_ax.axis('tight')
        table_ax.axis('off')
        error_dist = _share(mean_error)

        data = [ranks[:15], _fmt(mean_error[:15]), _fmt(error_dist[:15])]

        table = table_ax.table(cellText=data,
                               rowLabels=['Feature', 'Mean Error', 'Share'],
                               loc='center')
        table.auto_set_column_width(col=list(range(len(ranks[:15]))))

    plt.tight_layout()

    if return_figs:
        return figs
    else:
        plt.show()
        return None


def _create_lc(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    cmap,
    **kwargs
) -> LineCollection:
    """Creates a LineCollection.

    Args:
        x (np.ndarray): Data for x axis.
        y (np.ndarray): Data for y axis.
        labels (np.ndarray): Labels to color each point.
        cmap: Colormap.

    Returns:
        LineCollection: Returns a LineCollection.
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, array=labels, **kwargs)

    return lc


def _share(a: np.ndarray) -> np.ndarray:
    """Calculates the share for each element.

    Args:
        a (np.ndarray): Array.

    Returns:
        np.ndarray: Returns the share for each element.
    """
    a = minmax_scale(a)  # only positive values
    return a / np.sum(a)


def _fmt(a: T.Iterable) -> T.List[str]:
    """Formats each element in `a` into strings.

    Args:
        a (T.Iterable): T.Iterable.

    Returns:
        T.List[str]: Returns a list with the formatted elements.
    """
    return list(map(lambda a: f'{a:.3f}', a))


def plot_point_adjust_metrics(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    return_fig: bool = False
) -> T.Optional[Figure]:
    """Plots ``point_adjust_metrics``.

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground truth.
        return_fig (bool): Whether to return the figure (Default: False).

    Returns:
        T.Optional[Figure]: Returns the figure if return_fig=True.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> mlnext.plot_point_adjust_metrics(
        ...   y_hat=np.array([0, 1, 1, 0]), y=np.array([0, 1, 1, 1]))

    Expected result:

    .. image:: ../images/plot_point_adjust_metrics.png
        :scale: 100 %

    """

    df = point_adjust_metrics(y_hat=y_hat, y=y)
    df = df.reset_index().rename(columns={
        'index': 'K',
        **{
            col: f'{col} (AUC: {auc(df.index, df[col]) / 100:.2f})'
            for col in df
        }
    })
    dfm = df.melt('K', var_name='Metrics', value_name='Score')

    with sns.plotting_context('notebook'), sns.axes_style('darkgrid'):
        fig, ax = plt.subplots()
        g = sns.lineplot(
            x='K', y='Score', hue='Metrics',
            data=dfm, markers=True, marker='o',
            ax=ax
        )
        g.set(
            xlim=(0, 100),
            xticks=range(0, 101, 10),
            xticklabels=range(0, 101, 10)
        )

    return fig if return_fig else None
