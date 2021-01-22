"""
Class container for the Dynamic DSGE model
"""

__author__ = "Karl Naumann"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd


def _lhs(cons: float, x0: float) -> float:
    return (cons ** 2 / x0) ** 2


def _rhs(cons: float, gt: float, g: float = 1, r: float = 0) -> float:
    return 1 / (1 + r) - cons / ((gt - cons) * g)


def _bisection(x0: float, gt: float, gamma: float, r: float = 0,
               err: float = 1e-2) -> float:
    # Define the left and right hand sides of the equations
    diff = lambda a: _rhs(a, gt, gamma, r) - _lhs(a, x0)
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


class DynamicDSGE(object):
    """
        Class for the dynamic DSGE model where the representative household
        will invest money based on a feedback from the production
    """
    def __init__(self):
        self.params = dict(etaZ=0, sigmaZ=1.0, zbar=1, inflation=0.01,
                           interest=0.01, k0=0.0, xmin=0.1, xmax=0.9, theta=10,
                           c1=1.1, c2=1, c3=1, depreciation=0.2, gamma=1)
        self.path = None

    def set_params(self, **kwargs):
        """ Try to set all the parameters in the dictionary based on the keyword
        arguments that were given

        Parameters
        ----------
        kwargs  :   dict
        """

        for i, v in kwargs.items():
            try:
                self.params[i] = v
            except KeyError:
                pass

    def graph(self):
        pass

    def simulate(self, t_end: int, start: np.ndarray):
        """ Run a simulation of t_end steps in time for a given starting value

        Parameters
        ----------
        t_end   :   int
        start   :   np.ndarray
            starting values in the order: z, k, news, cons, labour, bond,
            feedback, wage, xi, r
        """
        x = np.empty((int(t_end), len(start)))
        x[0, :] = start
        for t in range(1, int(t_end)):
            x[t, :] = self._step(x[t - 1, :], self.params, t)
        cols = ['technology', 'capital', 'news', 'consumption',
                'labour', 'bond', 'feedback', 'wage', 'xi', 'coc']
        self.path = pd.DataFrame(x, columns=cols)

    def _step(self, x: np.ndarray, p: dict, t: float) -> np.ndarray:
        """ Single update step in the simulation

        Parameters
        ----------
        x   :   np.ndarray
        p   :   dict
        t   :   float

        Returns
        -------
        x   :   np.ndarray
        """
        # Starting variables
        z_, ks_, news_, cons_, labour_, bond_, feedback_, wage_, xi_, r_ = x

        # Random technology process
        rand = np.random.normal(0, p['sigmaZ'])
        xi = p['etaZ'] * xi_ + np.sqrt(1 - p['etaZ'] ** 2) * rand
        z = p['zbar'] * np.exp(xi)

        # Income and Investment
        income = wage_ * labour_ + bond_ / (1 + p['inflation']) + r_ * ks_
        m = np.tanh(p['theta'] * (news_ - p['threshold']))
        feedback = (m * (p['xmax'] - p['xmin']) + p['xmax'] + p['xmin']) / 2

        # Capital Markets
        ks = (1 - p['depreciation']) * ks_ + income * (1 - feedback)

        # Household decision variables
        x0 = z * np.sqrt(2 * ks)
        cons = _bisection(x0, feedback_ * income, p['gamma'], r_)
        labour = (cons ** 2) / (4 * ks * (z ** 2))
        bond = (feedback_ * income - cons) * (1 + r_)

        # Savings decision
        fake_news = 0  # np.random.normal(0, p['sigmaZ'])
        news = sum([
            p['c1'] * news_,
            p['c2'] * (cons / cons_ - 1),
            p['c3'] * (ks / ks_ - 1),
            fake_news,
        ])

        # Cost of production via mkt clearing and profit maximisation
        wage = 0.5 * cons / labour
        r = 2 * z * np.sqrt(labour / ks)

        return z, ks, news, cons, labour, bond, feedback, wage, xi, r


if __name__ == "__main__":
    pass
