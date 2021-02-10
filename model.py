"""
Functions for the calculation of the simulations to be run
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# SIMULATION FUNCTIONS


def bound(z: float, gti: float, k: float, p: dict):
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


def bisection(z: float, g: float, k: float, p: dict, eps: float = 1e-7):
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
    eps : float, default: 1e-5
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

    max_val = bound(z, k, p)

    # Adapt by precision to avoid asymptotic bounds
    edge = eps * 1e-2
    x = [edge, max_val / 2, max_val-edge]
    abs_lst = [abs(diff(i)) for i in x[:2]]

    # Conditions to stop: difference too small OR too close to the bound
    while all([min(abs_lst) >= eps, max_val - x[0] >= eps]):
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
    # Starting variables
    z_, c_, n_, b_, w_, k_, q_, g_, s_, s0, \
        news_, inc_, xiz_, xin_, crisis_ = x

    # Random technology process
    rand = np.random.normal(0, p['sigmaZ'])
    xiz = p['etaZ'] * xiz_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
    z = p['zbar'] * np.exp(xiz)

    # Observe "State of economy"
    g = g_
    signal = np.tanh(p['s_theta']*(s_ - s0))
    s = .5*((p['s_max']-p['s_min'])*signal + (p['s_max']+p['s_min']))

    # Determine Consumption
    c = bisection(z, g, k_, p)

    # Working hours via market clearing
    n = ((c / z) ** (-1 * p['mu']) - p['alpha'] * k_ ** (-1 * p['mu']))
    n = (n / (1 - p['alpha'])) ** (-1 / p['mu'])

    # Firm observes desired working hours, sets the wage accordingly
    rho = -1 * p['mu']
    temp = (p['alpha'] * k_ ** rho + (1 - p['alpha']) * n ** rho)
    temp = temp ** ((1 / rho) - 1)
    w = (1 - p['alpha']) * z * temp * (n ** (rho - 1))

    # Income
    income = w * n + (b_ + q_ * k_) / (1 + p['inflation'])

    # Investment & Bonds
    investment = income * (1 - g)
    b = (1 + p['interest']) * s * investment

    # Capital & Risky return
    k = (1 - p['depreciation']) * k_ + investment * (1 - s)
    q = p['alpha'] * z * temp * (k ** (rho - 1))

    # Retain previous news formula out of interest
    xin = np.random.normal(0, p['sigmaN'])
    info = p['n_cons']*(c/c_ - 1)
    step_news = p['n_persistence'] * news_ + (1 - p['n_persistence']) * info + xin
    news = np.tanh(p['n_theta'] * step_news)
    crisis = 0.5 * (1.1 + np.sign(n - k))
    s0 = (c/c_ - 1)

    return z, c, n, b, w, k, q, g, s, s0, news, income, xiz, xin, crisis

    x = 0
