"""
Functions for the calculation of the simulations to be run
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# STEADY STATE SIMULATIONS
def c_bound(z: float, k: float, p: dict):
    """Upper bound on the amount that can be consumed

    .. math:: c_t \leq z_t\cdotk_t\cdot\alpha^{-\frac{1}{\mu}}


    Parameters
    ----------
    z : float
        Level of productivity
    k : float
        Level of capital at t
    p : dict
        Parameters from simulation

    Returns
    -------
    bound : float
        Upper bound on consumption
    """
    return z * k * p['alpha'] ** (-1.0 / p['mu'])


def bisection(z: float, g: float, k: float, p: dict, precision: float = 1e-7):
    """ Determine the level of consumption using the bisection method

    .. math:: \frac{2\gamma}{1-\alpha} c_t -  G_t z_t \left(c_t^{-\mu } z_t^{\mu }\right)^{-\frac{\mu +1}{\mu }} \left(\frac{c_t^{-\mu } z_t^{\mu }-\alpha  k_t^{-\mu }}{1-\alpha }\right)^{\frac{2}{\mu }+1} = 0


    Parameters
    ----------
    z : float
        Level of productivity
    g : float
        Consumption rate (% of income consumed)
    k : float
        Level of capital at t
    p : dict
        Parameters from simulation
    precision : float, default: 1e-5
        Precision of the bisection solution

    Returns
    -------
    c : float
        level of consumption
    """

    # Pre-compute constants
    mu = p['mu']
    lhs_1 = 2.0 * p['gamma'] / (1.0 - p['alpha'])
    rhs_1 = g * z / ((1.0 - p['alpha']) ** (2.0 / mu + 1.0))
    rhs_2 = p['alpha'] * k ** (-1.0 * mu)

    # Minimisation target for the bisection
    def diff(c: float):
        r = z / c
        rhs = rhs_1 * (r ** (-1.0 - mu)) * ((r ** mu) - rhs_2) ** (2.0 / mu + 1.0)
        return c * lhs_1 - rhs

    max_val = c_bound(z, k, p)

    # Adapt by precision to avoid asymptotic bounds
    edge = precision * 1e-2
    x = [edge, max_val / 2.0, max_val - edge]
    abs_lst = [abs(diff(i)) for i in x[:2]]

    # Conditions to stop: difference too small OR too close to the bound
    while all([min(abs_lst) >= precision, max_val - x[0] >= precision]):
        test = np.sign([diff(i) for i in x])
        if test[0] == test[1]:
            x = [x[1], (x[1] + x[2]) / 2, x[2]]
        elif test[1] == test[2]:
            x = [x[0], (x[0] + x[1]) / 2, x[1]]

        abs_lst = [abs(diff(i)) for i in x[:2]]

    return x[np.argmin(abs_lst)]


def step(t: float, x: np.ndarray, p: dict, err: float):
    """Iteration of one step in the simulation

    Parameters
    ----------
    t : float
        Current timestep t
    x : np.ndarray
        state variables z, c, n, b, w, k, q, g, s, news, inc, xiz, xin
    p : dict
        Parameters from simulation
    err : float
        precision of the bisection method

    Returns
    -------
    bound : float
        Upper bound on consumption
    """
    # Starting variables
    z_, c_, n_, b_, w_, k_, q_, g_, s_, news_, inc_, xiz_, xin_ = x

    # Random technology process
    rand = np.random.normal(0.0, p['sigmaZ'])
    xiz = p['etaZ'] * xiz_ + np.sqrt(1.0 - p['etaZ'] ** 2.0) * rand
    z = p['zbar'] * np.exp(xiz)

    # Observe "State of economy"
    g = g_
    s = s_

    # Determine Consumption
    c = bisection(z, g, k_, p, precision=err)

    # Working hours via market clearing
    n = ((c / z) ** (-1.0 * p['mu']) - p['alpha'] * k_ ** (-1.0 * p['mu']))
    n = (n / (1.0 - p['alpha'])) ** (-1.0 / p['mu'])

    # Firm observes desired working hours, sets the wage accordingly
    rho = -1.0 * p['mu']
    temp = (p['alpha'] * k_ ** rho + (1.0 - p['alpha']) * n ** rho)
    temp = temp ** ((1.0 / rho) - 1.0)
    w = (1.0 - p['alpha']) * z * temp * (n ** (rho - 1.0))

    # Income
    income = w * n + (b_ + q_ * k_) / (1.0 + p['inflation'])

    # Investment & Bonds
    investment = income * (1.0 - g)
    b = (1.0 + p['interest']) * s * investment

    # Capital & Risky return
    k = (1.0 - p['depreciation']) * k_ + investment * (1.0 - s)
    q = p['alpha'] * z * temp * (k ** (rho - 1.0))

    # Retain previous news formula out of interest
    xin = np.random.normal(0.0, p['sigmaN'])
    info = p['n_cons'] * (c / c_ - 1.0)
    step_news = p['n_persistence'] * news_ + (
            1.0 - p['n_persistence']) * info + xin
    news = np.tanh(p['n_theta'] * step_news)

    return z, c, n, b, w, k, q, g, s, news, income, xiz, xin


def steady_state_simulate(start: np.ndarray, p: dict, t_max: float = 2e3, err: float = 1e-4):
    """ Complete a t_end period simulation of the whole system

    Parameters
    ----------
    start : np.ndarray
        starting variables z, c, n, b, w, k, q, g, s, news, inc, xiz, xin
    p : dict
        Parameters from simulation
    t_end : float
        Duration of the simulation

    Returns
    -------
    df : pd.DataFrame
        timeseries of the simulation results
    """
    prior, t, cond = start, 1, 1
    while cond:
        new = step(t, prior, p, 1e-5)
        t += 1
        cond = all([
            abs(new[5] - prior[5]) > err,
            any(np.isnan(new)) == False,
            t <= t_max])
        prior = new

    cols = ['z', 'c', 'n', 'b', 'w', 'k', 'q', 'g', 's', 'news', 'income',
            'xiz', 'xin']
    cols += ['utility']
    u = np.array([np.log(new[1]) - p['gamma'] * (new[2] ** 2)])
    new = np.hstack([new, u])
    if t >= t_max:
        print("Sim reached t_max")
    return pd.Series(new, index=cols)


def start_array(g, s, start_dict):
    # Set starting values and run simulation until the steady state
    start_dict['g'] = g
    start_dict['s'] = s
    return np.array([v for _, v in start_dict.items()])


def set_gs_range(gs_num):
    g = np.linspace(1e-3, 1 - 1e-3, gs_num)
    s = np.linspace(1e-3, 1 - 1e-3, gs_num)
    return g, s


def gs_steady_state(g_list, s_list, params, macro_vars, start_dict, T, err):
    empty_frame = pd.DataFrame(index=g_list, columns=s_list, dtype=float)
    res = {k: empty_frame.copy(deep=True) for k in macro_vars}
    for g, s in product(g_list, s_list):
        start = start_array(g, s, start_dict)
        x = steady_state_simulate(start, params, t_max=T, err=err)
        for k in res.keys():
            res[k].loc[g, s] = x.loc[k]
    return res


def sim_param_effect(param, param_range, gs_num, T, err, macro_vars,
                     params, start_dict):
    g_list, s_list = set_gs_range(gs_num)

    # Analysis of results
    results = {}

    # Go through parameters
    for val in param_range:
        params[param] = val
        results[val] = gs_steady_state(g_list, s_list, params, macro_vars,
                                       start_dict, T, err)

    return results


def find_cbar_range(dfs, top_cutoff: float = 0.90):
    vmin = min([df.min().min() for df in dfs])
    vmax = max([df.quantile(top_cutoff).quantile(top_cutoff) for df in dfs])
    return vmin, vmax


def plot_steady_state_effects(res, param, save=None, sup_tit=None,
                              n_lin: int = 20, top_cutoff: float = 0.9,
                              q_cutoff: float = 0.75, cmap: str = 'plasma'):
    
    keys = list(res.keys())
    nrow, ncol = len(res[keys[0]].keys()), len(keys)
    
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol)
    
    for i, var in enumerate(res[keys[0]].keys()):
        if len(keys) == 1:
            x = res[keys[0]][var]
            q = ax[i].contourf(x.columns, x.index, x, levels=n_lin)
            _ = plt.colorbar(q, ax=ax[i])
            ax[i].set_title("{}".format(var))
            ax[i].set_xlabel('s')
            ax[i].set_ylabel('g')
        else:
            dfs = [res[val][var] for val in keys]
            vmin, vmax = find_cbar_range(dfs, top_cutoff)
            if var == 'q':
                vmin, vmax = find_cbar_range(dfs, q_cutoff)
            kwargs = dict(levels=np.linspace(vmin, vmax, n_lin), cmap=cmap)
            for ii, val in enumerate(keys):
                x = res[val][var]
                q = ax[i, ii].contourf(x.columns, x.index, x, **kwargs)
                
                if ii == ncol - 1:
                    cbar = plt.colorbar(q, ax=ax[i, ii])
                    form = FormatStrFormatter('%0.1f')
                    cbar.ax.yaxis.set_major_formatter(form)

                ax[i, ii].set_title("{} {}={:.1e}".format(var, param, val))
                ax[i, ii].set_xlabel('s')
                ax[i, ii].set_ylabel('g')

    if sup_tit is not None:
        fig.suptitle(sup_tit, fontsize=16)

    fig.set_size_inches(4 * ncol, 4 * nrow)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save is not None:
        plt.savefig(save, bbox_inches='tight', format='pdf')
