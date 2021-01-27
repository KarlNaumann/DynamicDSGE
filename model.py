"""
Functions for the calculation of the simulations to be run
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# GRAPHING FUNCTIONS


def timeseries(ax, data, log: bool = True, title: str = ''):
    """ Function to graph a timeseries on a given axis

    Parameters
    ----------
    ax  :   matplotlib axes
    data  :   pd.DataFrame
    log     :   bool
    title   :   str
    """
    assert isinstance(data, pd.DataFrame), "Graph dataframe only"

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


def simulation_graph(groups: dict):
    """ Plot the given time series in groups

    Parameters
    ----------
    groups  :   dict

    Returns
    --------
    axs     :   dict
    """
    fig, ax = plt.subplots(ncols=1, nrows=len(list(groups.keys())))
    axs = {}
    for i, k in enumerate(groups.keys()):
        timeseries(ax[i], *groups[k], k)
        axs[k] = ax[i]
    plt.tight_layout()
    plt.show(block=False)


# SIMULATION FUNCTIONS


def bisection_generalCB(z: float, income: float, k: float, gt_: float,
                        ft_: float, p: dict, err: float = 1e-5) -> float:
    """ Bi-section method for the general Cobb-Douglas production function

    Parameters
    ----------
    z   :    float
    income  :   float
    k   :   float
    gt_     :   float
    ft_     :   float
    p   :   dict
    err     :   float (default 1e-5)

    Returns
    ----------
    cons    :   float
    """
    def lhs(c: float, z: float, k: float, a: float, g: float = 1,
            r: float = 0) -> float:
        pt1 = 2 * g * (1+r) / (1 - a)
        pt2 = (c / (z * k**a)) ** (2 / (1-a))
        return pt1 * pt2

    def rhs(cons: float, gti: float, ft: float) -> float:
        return 1 - (ft * cons) / (gti - cons)

    def diff(c: float) -> float:
        lhs_args = [c, z, k, p['alpha'], p['gamma'], p['interest']]
        rhs(c, gt_ * income, ft_) - lhs(*lhs_args)

    # Initial guess at the next options for
    max_val = gt_ * income / (1+ft_)
    x = [0, max_val / 2, max_val]
    abs_lst = [abs(diff(i)) for i in x[:2]]

    while min(abs_lst) >= err:
        test = np.sign([diff(i) for i in x])

        if test[0] == test[1]:
            x = [x[1], (x[1] + x[2]) / 2, x[2]]
        elif test[1] == test[2]:
            x = [x[0], (x[0] + x[1]) / 2, x[1]]

        abs_lst = [abs(diff(i)) for i in x[:2]]

    return x[np.argmin(abs_lst)]


def bisection_CES(z: float, income: float, k: float, gt_: float, ft_: float,
                  p: dict, err: float = 1e-5) -> float:
    """ Bi-section method for the CES production function

    Parameters
    ----------
    z   :    float
    income  :   float
    k   :   float
    gt_     :   float
    ft_     :   float
    p   :   dict
    err     :   float (default 1e-5)

    Returns
    ----------
    cons    :   float
    """

    def lhs(cons: float, z: float, k: float, alpha: float, gamma: float,
            r: float, rho: float) -> float:
        pt1 = 2 * gamma * (1+r) / ((z ** 2) * (1 - alpha) ** (2 / rho))
        pt2 = cons ** 2
        pt3 = (1 - alpha*(z * k / cons) ** rho) ** ((2 - rho) / rho)
        return pt1 * pt2 * pt3

    def rhs(cons: float, gti: float, ft: float) -> float:
        return 1 - (ft * cons) / (gti - cons)

    def diff(c: float) -> float:
        lhs_args = [c, z, k, p['alpha'], p['gamma'], p['interest'], p['rho']]
        rhs(c, gt_ * income, ft_) - lhs(*lhs_args)

    # Initial guess at the next options
    guess = np.min([gt_*income, k * z * (1 / p['alpha']) ** (-1 / p['rho'])])
    x = [0, guess / 2, guess]
    abs_lst = [abs(diff(i)) for i in x[:2]]

    # Apply bi-section method
    while min(abs_lst) >= err:
        test = np.sign([diff(i) for i in x])

        if test[0] == test[1]:
            x = [x[1], (x[1] + x[2]) / 2, x[2]]
        elif test[1] == test[2]:
            x = [x[0], (x[0] + x[1]) / 2, x[1]]

        abs_lst = [abs(diff(i)) for i in x[:2]]

    return x[np.argmin(abs_lst)]


def step(t: float, x_: np.ndarray, p: dict):
    """ Iterate through one step of the economy

    Parameters
    ----------
    t   :   float
    x_  :   np.ndarray
    p   :   dict

    Returns
    ----------
    x   :   np.ndarray
    """
    # Starting variables
    z_, c_, n_, b_, w_, k_, q_, gt_, ft_, news_, inc_, xiz_, xin_ = x_

    # Random technology process
    rand = np.random.normal(0, p['sigmaZ'])
    xiz = p['etaZ'] * xiz_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
    z = p['zbar'] * np.exp(xiz)

    # Income and Investment
    income = (w_ * n_ + b_ + q_ * k_) / (1 + p['inflation'])

    # Capital Markets
    k = (1 - p['depreciation']) * k_ + income * (1 - gt_)

    # Household decision
    c = bisection_CES(z, income, k, gt_, ft_, p)
    n = (c ** 2) / (4 * k * (z ** 2))
    b = (gt_ * income - c) * (1 + p['interest'])

    # Firm decisions (CES)
    temp = (p['alpha'] * k ** p['rho'] + (1 - p['alpha']) * n ** p['rho'])
    temp = temp ** ((1 / p['rho']) - 1)
    w = (1 - p['alpha']) * z * temp * (n ** (p['rho'] - 1))
    q = p['alpha'] * z * temp * (k ** (p['rho'] - 1))

    # News
    xin = np.random.normal(0, p['sigmaN'])
    info = p['n_cons']*(c/c_ - 1)
    temp = p['n_persistence'] * news_ + (1 - p['n_persistence']) * info + xin
    news = np.tanh(p['n_theta'] * temp)

    if t > 300 and t < 400:
        if p['shock'] == -1:
            news = -1
        elif p['shock'] == 1:
            news = 1

    # Household modifiers
    gt = 0.5 * (p['g_max'] + p['g_min'] - news * (p['g_max'] - p['g_min']))
    ft = 0.5 * (p['f_max'] + p['f_min'] - news * (p['f_max'] - p['f_min']))

    return z, c, n, b, w, k, q, gt, ft, news, income, xiz, xin


def simulate(start: np.ndarray, p: dict, t_end: float = 1e3):
    """ Function to simulate the economy

    Parameters
    ----------
    start   :   np.ndarray
    p   :   dict
    t_end   :   float (default 1e3)

    Returns
    ----------
    df  :   pd.DataFrame
    """
    x = np.empty((int(t_end), len(start)))
    x[0, :] = start
    for t in range(1, int(t_end)):
        x[t, :] = step(t, x[t - 1, :], p)
    cols = ['z', 'c', 'n', 'b', 'w', 'k', 'q', 'gt',
            'ft', 'news', 'income', 'xiz', 'xin']
    df = pd.DataFrame(x, columns=cols)
    df.loc[:, 'inv'] = 100*(1-df.loc[:, 'gt'])
    df.loc[:, 'bc'] = df.b / df.c
    return df
