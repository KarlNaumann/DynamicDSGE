"""
Functions for the calculation of the simulations to be run
"""

__author__ = "Karl Naumann & Federico Morelli"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd


def lhs(cons: float, x0: float) -> float:
    """ Left-hand side of the household optimality condition that determines
    consumption

    Parameters
    ----------
    cons    :   float
        Consumption level
    x0  :   float
        Product 2 * z^2 * k

    Returns
    -------
    lhs     :   float
    """
    return (cons ** 2 / x0) ** 2


def rhs(cons: float, gt: float, gamma: float = 1, r: float = 0) -> float:
    """ Right-hand side of the household optimality condition that determines
    consumption

    Parameters
    ----------
    cons    :   float
        consumption level
    gt  :   float
        effective income G * I
    gamma   :   float
        disutility of labour
    r     :   float
        interest rate

    Returns
    -------
    rhs     :   float

    """

    # Old version: return (2 + gt / (cons - gt)) / (gamma * (1+r))

    return 1 / (1 + r) - cons / ((gt - cons) * gamma)


def bisection(x0: float, gt: float, gamma: float, r: float = 0,
              err: float = 1e-2) -> float:
    """ Numerical solution to the Households optimisation problem by means of
    a bisection method. Solution is guaranteed to exist.

    Parameters
    ----------
    x0  :   float
        Product 2 * z^2 * k
    gt  :   float
        effective income G * I
    gamma   :   float
        disutility of labour
    r     :   float
        interest rate
    err :   float
        precision to the optimal solution

    Returns
    -------
    cons   :   float
        household optimal consumption level
    """

    # Define the left and right hand sides of the equations
    diff = lambda a: rhs(a, gt, gamma, r) - lhs(a, x0)

    # Initial guess at the next options for
    x = [0, gt / 4, gt / 2]
    abs_lst = [abs(diff(i)) for i in x[:2]]

    while min(abs_lst) >= err:
        test = np.sign([diff(i) for i in x])

        if test[0] == test[1]:
            x = [x[1], (x[1] + x[2]) / 2, x[2]]
        elif test[1] == test[2]:
            x = [x[0], (x[0] + x[1]) / 2, x[1]]

        abs_lst = [abs(diff(i)) for i in x[:2]]

    return x[np.argmin(abs_lst)]


def hh_feedback(x: float, x0: float, xmin: float = 0, xmax: float = 0.7,
                a: float = 10):
    """ Household feedback function G, to determine what proportion of their
    income to invest as capital supply

    Parameters
    ----------
    x   :   float
    x0  :   float
        Inflection level
    xmin    :   float
        Minimal savings rate
    xmax    :   float
        Maximal savings rate
    a   :   float
        Tanh multiplier

    Returns
    -------
    g   :   float
    """
    #TODO Check if functional form still makes sense
    return 1 - 0.5 * (np.tanh(a * (x - x0)) * (xmax - xmin) + xmax + xmin)


def firm_expectation():
    return 0


def simulate(start: np.ndarray, p: dict, t_end: float = 1e3):
    """ Simulate a timeseries of the model realisations

    Parameters
    ----------
    start   :   np.ndarray
        starting values in the order
        z, k, ks, kd, s, cons, labour, bond, feedback, wage, xi
    p   :   dict
        dictionary of float values for the parameters that should include
        etaZ, sigmaZ, zbar, inflation, interest, k0, xmin, xmax, theta, c1, c2,
        depreciation, gamma
    t   :   float
        total time of the simulation

    Returns
    -------
    df  :   pd.DataFrame
        Timeseries of the individual variables in the model
    """

    x = np.empty((int(t_end), len(start)))
    x[0, :] = start
    for t in range(1, int(t_end)):
        x[t, :] = step(t, x[t-1, :], p)

    vars = ['technology', 'capital', 'ks', 'kd', 'sentiment', 'consumption',
            'labour', 'bond', 'feedback', 'wage', 'xi']
    return pd.DataFrame(x, columns=vars)


def step(t: float, x: np.ndarray, p: dict):
    """ Calculation of one time-step for the whole model

    Parameters
    ----------
    t   :   float
        current time (unused)
    x   :   np.ndarray
        np.ndarray of prior (t-1) realisations in the order
        z, k, cons, labour, bond, ks_dot, kd_dot, feedback, wage, xi
    p   :   dict
        dict of the parameters that are used

    Returns
    -------
    x_new   :   np.ndarray
        new realisation in order
        z, k, cons, labour, bond, ks_dot, kd_dot, feedback, wage, xi
    """
    #
    z_, k_, ks_, kd_, s_, cons_, labour_, bond_, feedback_, wage_, xi_ = x

    # Random technology process
    rand = np.random.normal(0, p['sigmaZ'])
    xi = p['etaZ'] * xi_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
    z = p['zbar'] * np.exp(xi)

    # Income and Investment
    income = wage_ * labour_ + bond_ / (1 + p['inflation'])
    #TODO Update the feedback to be a function of the news
    feedback = hh_feedback(k_, p['k0'], p['xmin'], p['xmax'], p['theta'])
    ks = ks_ + income * (1 - feedback)

    # Capital Markets
    s_dot = firm_expectation()
    kd = kd_ + p['c1'] * s_dot + p['c2'] * (s_ + s_dot)

    k = min([kd, ks]) - p['depreciation'] * k_

    x0 = z * np.sqrt(2 * k)

    # Household decision variables
    cons = bisection(x0, feedback * income, p['gamma'], p['interest'])
    labour = (cons ** 2) / (4 * k * (z ** 2))
    wage = 0.5 * cons / labour
    bond = (feedback * income - cons) * (1 + p['interest'])

    return z, k, ks, kd, s_ + s_dot, cons, labour, bond, feedback, wage, xi
