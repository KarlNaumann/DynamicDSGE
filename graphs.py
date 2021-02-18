"""
Graphing helper functions
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"


def timeseries(ax, data, log: bool = True, title: str = ''):
    """ Function to graph a timeseries on a given axis
    Parameters
    ----------
    ax  :   matplotlib axes
    data  :   pd.DataFrame
    log     :   bool
    title   :   str
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    for c in data.columns:
        ax.plot(data.loc[:, c], label=c)
    if data.shape[1] > 1:
        ax.legend()
    try:
        t = ' '.join(data.columns)
    except AttributeError:
        t = ''

    if log:
        ax.set_yscale('log')
        t = 'log ' + t
    if title == '':
        ax.set_title(t)
    else:
        ax.set_title(title)


def simulation_graph(groups: dict, size: tuple = (3, 10), save=''):
    """ Plot the given time series in groups
    
    Parameters
    ----------
    groups  :   dict
        different desired graphs index = title, content = list of 
        data and log scale bool

    Returns
    --------
    axs     :   dict
        each axis object indexed by the subplot title
    """
    fig, ax = plt.subplots(ncols=1, nrows=len(list(groups.keys())))
    fig.set_size_inches(size)
    axs = {}
    for i, k in enumerate(groups.keys()):
        timeseries(ax[i], *groups[k], k)
        axs[k] = ax[i]
    plt.tight_layout()
    plt.show(block=False)
    if save != '':
        plt.savefig(save, bbox_inches='tight', format='pdf')
    return axs