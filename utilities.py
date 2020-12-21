import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import rc
from matplotlib.cm import get_cmap


def page_width():
    return 5.95114


def plot_settings(cycles: bool = True):
    """ Set the parameters for plotting such that they are consistent across
    the different models
    """
    rc('figure', figsize=(page_width(), 6))

    # General Font settings
    x = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
    rc('text.latex', preamble=x)
    rc('text', usetex=True)
    rc('font', **{'family': 'serif'})

    # Font sizes
    base = 12
    rc('axes', titlesize=base)
    rc('legend', fontsize=base - 2)
    rc('axes', labelsize=base - 2)

    # Axis styles
    cmap = get_cmap('tab10')
    cycles = cycler('color', cmap([0.05, 0.15, 0.25, 0.35]))
    if cycles:
        cycles += cycler('linestyle', ['-', '--', ':', '-.'])
    else:
        cycles += cycler('linestyle', 4 * ['-'])
    rc('axes', prop_cycle=cycles)


def time_series_plot(data, ax, xtxt: str = '', ytxt: str = ''):
    """ Generate a timeseries graph on the axes for each column in the
    given dataframe

    Parameters
    ----------
    df  :   pd.DataFrame
    ax  :   matplotlib axes object

    Returns
    ----------
    ax  :   matplotlib axes object
    """

    if isinstance(data, pd.DataFrame):
        for series in data.columns:
            ax.plot(data.loc[:, series], label=series)
        if len(data.columns) > 1:
            ncol = len(data.columns) if len(data.columns) < 3 else 2
            ax.legend(ncol=ncol)
    elif isinstance(data, pd.Series):
        ax.plot(data, label=data.name)
    else:
        raise TypeError

    if xtxt == '':
        try:
            ax.set_xlabel(' '.join(data.index.names))
        except KeyError:
            try:
                ax.set_xlabel(data.index.name)
            except KeyError:
                pass
    else:
        ax.set_xlabel(xtxt)

    ax.set_ylabel(ytxt)

    ax.set_xlim(data.index[0], data.index[-1])
    try:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 4))
    except AttributeError:
        pass
    # ax.minorticks_on()

    return ax


def histogram(series: pd.Series, ax, win: bool = True, bins: int = 20,
              xtxt: str = '', ytxt: str = 'Proportion'):
    """ Function to generically plot a histogram

    Parameters
    ----------
    Series  :   pd.Series
    ax  :   matplotlib axis object
    win :   bool
    xtxt    :   str
    ytxt    :   str
    """

    n, bins = np.histogram(series, bins=bins)
    n = n / series.shape[0]

    ax.hist(bins[:-1], bins, weights=n)
    ax.set_xlabel(xtxt)
    ax.set_ylabel(ytxt)
