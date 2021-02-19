"""
Main controlling script for the simulations
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

import pandas as pd
import numpy as np

# SIMULATION FUNCTIONS
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
    return z * k * p['alpha'] ** (-1 / p['mu'])


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
    lhs_1 = 2 * p['gamma'] / (1 - p['alpha'])
    rhs_1 = g * z / ((1 - p['alpha']) ** (2 / mu + 1))
    rhs_2 = p['alpha'] * k ** (-1 * mu)

    # Minimisation target for the bisection
    def diff(c: float):
        r = z / c
        rhs = rhs_1 * (r ** (-1 - mu)) * ((r ** mu) - rhs_2) ** (2 / mu + 1)
        return c * lhs_1 - rhs

    max_val = c_bound(z, k, p)

    # Adapt by precision to avoid asymptotic bounds
    edge = precision * 1e-2
    x = [edge, max_val / 2, max_val - edge]
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


def default_step(t: float, x: np.ndarray, p: dict, err: float):
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
    z_, c_, n_, b_, w_, k_, k__, q_, g_, s_, news_, _, xiz_, _ = x

    # Random technology process
    rand = np.random.normal(0, p['sigmaZ'])
    xiz = p['etaZ'] * xiz_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
    z = p['zbar'] * np.exp(xiz)

    # Observe "State of economy"
    g = g_
    signal = np.tanh(p['s_theta'] * (s_ - p['news_spread'] * news_))
    s = 0.5 * ((p['s_max'] - p['s_min']) * signal + p['s_max'] + p['s_min'])

    # Determine Consumption
    c = bisection(z, g, k_, p, precision=err)

    # Working hours via market clearing
    n = (c / z) ** (-1 * p['mu']) - p['alpha'] * k_ ** (-1 * p['mu'])
    n = (n / (1 - p['alpha'])) ** (-1 / p['mu'])

    # Firm observes desired working hours, sets the wage accordingly
    rho = -1 * p['mu']
    temp = p['alpha'] * k_ ** rho + (1 - p['alpha']) * n ** rho
    temp = temp ** ((1 / rho) - 1)
    w = (1 - p['alpha']) * z * temp * (n ** (rho - 1))

    # Income
    income = w * n + (b_ + q_ * k__) / (1 + p['inflation'])
    cash = g * income - c

    # Investment & Bonds
    investment = income * (1 - g)
    b = (1 + p['interest']) * s * investment

    # Capital & Risky return
    k = (1 - p['depreciation']) * k_ + investment * (1 - s)
    q = p['alpha'] * z * temp * (k_ ** (rho - 1))

    # Signals to the household investor
    #theta_c = (p['s_interval'] * c_) ** -1 
    #info_c = np.tanh(theta_c * (c - c_))
    #theta_r = (p['interest'] * p['s_interval']) ** -1
    #info_r = np.tanh(theta_r * (q - p['interest']))
    
    info_c = c / c_ - 1
    #info_c = (c - c_) / (c + c_)
    info_r = (q - p['interest']) / (q + p['interest'])
    news = p['s_c_weight'] * info_c + (1 - p['s_c_weight']) * info_r

    return z, c, n, b, w, k, k_, q, g, s, news, income, xiz, cash


def default_params():
    """ Return the default parameters for the dynamic DSGE model 
    
    Returns
    -------
    params : dict
    """
    return {
        # Technology parameters
        'etaZ': 0.2, 'sigmaZ': 0.2, 'zbar': 1.0,
        # Economic parameters
        'inflation': 0.01, 'interest': 0.01, 'depreciation': 0.01,
        # Sentiment Parameters
        's_min': 1e-4, 's_max': 1-1e-4,'s_theta':5.0,
        # Signal Parameters
        's_c_weight':.9, 's_interval':.1, 'news_spread':.7,
        # Household and Production parameters
        'gamma': 1.0, 'alpha': 0.33, 'mu': 12.32}


def gen_params(**kwargs):
    """ Generate parameters leaving the rest as defaults. 
    
    Parameters
    ----------
    name, value pairs for desired parameter changes

    Returns
    -------
    parameters : dictionary of parameters
    """
    params = default_params()
    for arg, val in kwargs.items():
        assert arg in params.keys(), "{} is invalid".format(arg)
        params[arg] = val
    return params


def default_start():
    return dict(z=1.0, c=1.0, n=1.0, b=1.0, 
                w=1.0, k=1.1, k_=0.0, q=0.0, 
                g=0.7, s=0.5, news=1.0, 
                income=1.0, xiz=0.0, cash=0.0)


def gen_start(**kwargs):
    """ Generate parameters leaving the rest as defaults. 
    
    Parameters
    ----------
    name, value pairs for desired parameter changes

    Returns
    -------
    parameters : dictionary of parameters
    """
    start = default_start()
    for arg, val in kwargs.items():
        assert arg in start.keys(), "{} is invalid".format(arg)
        start[arg] = val
    return start


def simulate(start: dict=None, p: dict=None, step_func=default_step, 
             t_end: float = 1e3, err: float = 1e-4):
    """ Complete a t_end period simulation of the whole system

    Parameters
    ----------
    start : dict
        starting variables z, c, n, b, w, k, q, g, s, news, inc, xiz, xin
    p : dict
        Parameters from simulation
    setp_func   :   functionnews_spread
        Function with which to do the simulation
    t_end : float
        Duration of the simulation
    err     :   float
        precision of the bisection method

    Returns
    -------
    df : pd.DataFrame
        timeseries of the simulation results
    """

    p = p if p is not None else default_params()
    start = start if start is not None else default_start()

    init = [v for _, v in start.items()]

    x = np.empty((int(t_end), len(start)))
    x[0, :] = init
    for t in range(1, int(t_end)):
        x[t, :] = step_func(t, x[t - 1, :], p, err)
        if any([x[t, 1] < err, x[t, 2] < err, x[t, 5] < err]):  # c, n, k
            break
    x = x[:t+10, :]
    return pd.DataFrame(x, columns=start.keys())
