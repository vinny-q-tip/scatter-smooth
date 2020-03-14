import datetime
from typing import Union, List, Tuple, Any
import numpy as np
import logging
from pandas._libs.tslibs.timestamps import Timestamp
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, datestr2num
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle


def plot_scatterdata(xs: List[Union[float, Timestamp, datetime.date]],
                     ys: List[float],
                     smoother: str = None,
                     avoid_dups_for_smooth: bool = True,
                     degree: int = 1,
                     low_frac: float = 0.66,
                     low_it: int = 3,
                     low_delta: float = 0.0,
                     spl_smooth: float = 0.0,
                     spl_weight: List[float] = None,
                     y_limit: Tuple[float, float] = None,
                     x_ticks: List[float] = None,
                     x_label: str = None,
                     y_label: str = None,
                     title: str = None,
                     figsize: Tuple[int, int] = None,
                     sc_marker_size: int = 8,
                     sc_marker: Union[str, MarkerStyle] = None,
                     sc_color: Union[str, Any] = None,
                     sc_cmap: Colormap = None,
                     sc_norm: Normalize = None,
                     sc_vmin: float = None,
                     sc_vmax: float = None,
                     sc_alpha: float = None,
                     sc_linewidths: int = None,
                     sc_edgecolors: Union[str, Any] = None,
                     pl_alpha: float = None,
                     pl_color: Union[str, Any] = 'black',
                     pl_linestyle: Union[str, Any] = None,
                     pl_linewidth: int = 2,
                     pl_label: str = None):
    """
    A scatter plot of *ys* vs *xs* with multiple options for smoothing.

    Parameters
    ----------
    xs, ys :
        The data points. *xs* can be numbers or dates.

    smoother :
        The method with which the smoothing curve is computed. Possible values:

        - None (default): No smoothing curve is computed.
        - 'linear': Linear regression.
        - 'poly': Polynomial regression.
        - 'lowess': Locally weighted scatterplot smoothing.
        - 'splines': Smoothing splines.

    avoid_dups_for_smooth :
        If True, duplicates are avoided in the coputation of the smoothing curve by computing the average.

    degree :
        The degree of the polynomials. Only used in 'poly' and 'splines' smoothing.

    low_fraq :
        Between 0 and 1. The fraction of the data used when estimating each y-value in the lowess smoothing.

    low_it :
        The number of residual-based reweightings to perform in the lowess smoothing.

    low_delta :
        Distance within which to use linear-interpolation instead of weighted regression in the lowess smoothing.

    spl_smooth :
        Positive smoothing factor in the spline smoothing. If 0, spline will interpolate through all data points.

    spl_weight :
        Weights for spline fitting. Must be positive. If None (default), weights are all equal.

    y_lim, x_ticks, y_label, x_label, title, figsize :
        Parameters used to customize the matplotlib plot

    sc_marker_size, sc_marker, sc_color, sc_cmap, sc_norm,
     sc_vmin, sc_vmax, sc_alpha, sc_linewidths, sc_edgecolors :
        Parameters used to customize the scatter plot part of the plot.

    pl_alpha, pl_color, pl_linestyle, pl_linewidth, pl_label :
        Parameters used to customize the curve plot part of the plot.
    """

    # transform time data
    is_date_data = type(xs[0]) == Timestamp or type(xs[0]) == datetime.date
    if is_date_data:
        xs = [datestr2num(str(x)) for x in xs]

    # make sure xs is sorted
    if not all(xs[i] <= xs[i+1] for i in range(len(xs)-1)):
        xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))

    # avoid duplicates by averaging
    xs_scatter = xs
    ys_scatter = ys
    if not all(xs[i] < xs[i+1] for i in range(len(xs)-1)):
        if avoid_dups_for_smooth or smoother == 'splines':
            ys_per_x = {}
            for i in range(len(xs)):
                ys_per_x.setdefault(xs[i], []).append(ys[i])
            xs, ys = zip(*sorted([(x, sum(ys_per_x[x]) / len(ys_per_x[x]))
                                  for x in ys_per_x], key=lambda t: t[0]))

    if smoother == 'lowess':
        low_out = lowess(ys, xs, frac=low_frac, it=low_it, delta=low_delta)
        _, ys_smooth = list(map(list, zip(*low_out)))

    elif smoother == 'linear':
        p = np.polyfit(xs, ys, deg=1)
        ys_smooth = np.polyval(p, xs)

    elif smoother == 'splines':
        spl = UnivariateSpline(xs, ys, w=spl_weight, k=degree, s=spl_smooth)
        ys_smooth = spl(xs)

    elif smoother == 'polyfit':
        p = np.polyfit(xs, ys, deg=degree)
        ys_smooth = np.polyval(p, xs)

    elif smoother is not None:
        logging.error(f"Unknown smoother: {smoother}")
        return

    if figsize:
        plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.scatter(xs_scatter, ys_scatter, s=sc_marker_size, alpha=sc_alpha, c=sc_color, marker=sc_marker, cmap=sc_cmap,
               norm=sc_norm, vmin=sc_vmin, vmax=sc_vmax, linewidths=sc_linewidths, edgecolors=sc_edgecolors)
    if smoother is not None:
        ax.plot(xs, ys_smooth, alpha=pl_alpha, color=pl_color, linestyle=pl_linestyle, linewidth=pl_linewidth,
                label=pl_label)

    if x_ticks:
        if is_date_data:
            x_ticks = [datestr2num(str(datetime.date(year, 1, 1)))
                       for year in x_ticks]
        ax.set_xticks(x_ticks)

    # handle dates on x-axis
    @plt.FuncFormatter
    def fake_dates(x, pos):
        return num2date(x).strftime('%Y')
    if is_date_data:
        ax.xaxis.set_major_formatter(fake_dates)

    if y_limit:
        ax.set_ylim(y_limit)
    if y_label:
        ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    if title:
        ax.set_title(title)
    ax.tick_params(labelrotation=0)
