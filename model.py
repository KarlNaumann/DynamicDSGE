"""
Main controlling script for the simulations
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd


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

    Returns
    -------
    bound : float
        Upper bound on consumption
    """
    # Variables for the simulation
    z_, xiz_, c_, n_, b_, w_, k_, m_, k__, q_, g_, s_, s0_, mu_, sig2_, news_, c0_, sharpe_ = x

    # Random technology process
    if p['steady_state']:
        rand = np.random.normal(0, p['sigmaZ'])
    else:
        rand = np.random.normal(0, p['sigmaZ'])
    xiz = p['etaZ'] * xiz_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
    z = p['zbar'] * np.exp(xiz)

    # Determine Consumption
    c = bisection(z, g_, k_, p)

    # Working hours via market clearing
    n = ((c / z) ** (-1 * p['mu']) - p['alpha'] * k_ ** (-1 * p['mu']))
    n = (n / (1 - p['alpha'])) ** (-1 / p['mu'])

    # Firm observes desired working hours, sets the wage accordingly
    rho = -1 * p['mu']
    temp = (p['alpha'] * k_ ** rho + (1 - p['alpha']) * n ** rho)
    temp = temp ** ((1 / rho) - 1)
    w = (1 - p['alpha']) * z * temp * (n ** (rho - 1))

    # Income
    income = w * n + (b_ + q_ * k__) / (1 + p['inflation'])
    m = g_ * income - c

    # Investment & Bonds
    investment = income * (1 - g_)
    b = (1 + p['interest']) * s_ * investment

    # Capital & Risky return

    k = (1 - p['depreciation']) * k_ + investment * (1 - s_)
    temp = (p['alpha'] * k ** rho + (1 - p['alpha']) * n ** rho)
    temp = temp ** ((1 / rho) - 1)
    q = p['alpha'] * z * temp * k ** (rho - 1)

    # Returns to the household's portfolio
    # w_b = b_ / (b_ + k__)
    # port_ret = w_b * p['interest'] + (1 - w_b) * q_

    w_b = k_ / (b_ + k_)
    port_ret = q_  # k_ / (b_+k_) * q_

    # Expectations are based on EWMA (returns and volatility)

    mu = p['memory_1'] * mu_ + (1 - p['memory_1']) * port_ret
    sig2 = p['memory_1'] * sig2_ + (1 - p['memory_1']) * (port_ret - mu_) ** 2

    sharpe = (p['interest'] - mu) / np.sqrt(sig2)

    c0 = p['memory_2'] * c0_ + (1 - p['memory_2']) * ((c_ / c) - 1)

    s0 = sharpe
    news = p['ratio'] * s0 + (1 - p['ratio']) * (
        c0)  # + np.random.uniform(-p['ex_news'],p['ex_news'])

    # Risk-weighted excess returns are the signal
    if p['steady_state']:
        s0 = 1e9
        # s0 large = good signal

    # Decision on Spending and Investment Allocation
    g = g_

    financial_risk = np.random.beta(a=p['q_shock'], b=1)
    q = q * financial_risk

    s = 0.5 * ((p['s_max'] - p['s_min']) * np.tanh(p['s_theta'] * news) + p[
        's_max'] + p['s_min'])

    return z, xiz, c, n, b, w, k, m, k_, q, g, s, s0, mu, sig2, news, c0, sharpe


def default_params():
    """ Return the default parameters for the dynamic DSGE model

    Returns
    -------
    params : dict
    """
    return {
        # Noise Parameters
        'etaZ': 0.5, 'zbar': 1.0,
        # Empirical Parameters
        'inflation': 0.0015, 'interest': 0.001, 'depreciation': 0.1,
        # Quasi-fixed parameters
        's_min': 0, 's_max': 1, 'memory_1': 0.98, 'memory_2': .5, 'gamma': 1.0,
        'alpha': 0.33, 'mu': 7.32,
        # Variable parameters
        'sigmaZ': .8, 'q_shock': 10, 's_theta': 10, 'ratio': .5,
        'steady_state': False}


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
    return dict(z=1, xiz=1, c=1, n=1, b=1, w=1, k=.1, m=0.0, k_=0.0, q=0.02,
                g=0.7, s=0.5, s0=0.0, mu=0.01, sig2=1e-5, news=0, c0=0,
                income=0)


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


def simulate(start: dict = None, p: dict = None, step_func=step,
             t_end: float = 1e3, err: float = 1e-4):
    """ Complete a t_end period simulation of the whole system

    Parameters
    ----------
    start : dict
        starting variables z, c, n, b, w, k, q, g, s, news, inc, xiz, xin
    p : dict
        Parameters from simulation
    step_func   :   function for a single step in the model
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
    x = np.empty((int(t_end), len(start)))

    x[0, :] = [v for _, v in start.items()]
    for t in range(1, int(t_end)):
        x[t, :] = step_func(t, x[t - 1, :], p, err)

    return pd.DataFrame(x[:t + 1, :], columns=start.keys())
