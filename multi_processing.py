"""
Module Docstring
"""

__author__ = "Karl Naumann"
__version__ = "0.1.0"
__license__ = "MIT"

import multiprocessing as mp
from itertools import product

import model as model
import numpy as np
import pandas as pd
import pickle as pickle
import datetime as dt
import time as time


def generate_parameters_2_variations(x: dict):
    """ Generate a list of dictionaries with different parameter settings

    Parameters
    ----------
    x   :   dict

    Returns
    ----------
    params   :   list
    """
    keys = tuple(x.keys())
    prods = product(*[v for _, v in x.items()])
    return [{keys[i]: p[i] for i in [0, 1]} for p in prods]


def main_call(vars: dict):
    T = int(1e6)
    np.random.seed(40)
    params = model.gen_params(**vars)
    df = model.simulate(p=params, t_end=T)
    crises = 0.5 * (1.0 + np.sign(df.n - df.k))
    return pd.DataFrame([sum(crises) / T, sum(-df.s0) / T],
                        index=['proportion', 'sharpe'],
                        columns=[tuple([v for _, v in vars.items()])],
                        dtype=float).T


def convert_results(df: pd.DataFrame, vars:list):
    df.index = pd.MultiIndex.from_tuples(df.index.to_list(), names=vars)
    return {c: df.loc[:, c].unstack(level=-1) for c in df.columns}


def main(vars):
    args = generate_parameters_2_variations(vars)
    # Run the multiprocessing
    pool = mp.Pool(processes=max([1, mp.cpu_count() - 5]))
    results = pool.map(main_call, args)
    pool.close()
    pool.join()
    return convert_results(pd.concat(results), list(vars.keys()))


if __name__ == "__main__":
    # Generate Args
    k = 20

    """
    start = time.time()
    x = dict(q_shock=np.logspace(-1, 1.3, k), ratio=np.linspace(0, 1, k))
    file = open('data_shock_v_ratio.dict', 'wb')
    v = main(x)
    pickle.dump(v, file)
    file.close()
    print("Completed in {}".format(dt.timedelta(seconds=time.time()-start)))
    """
    start = time.time()
    x = dict(q_shock=np.logspace(-1, 1.3, k), sigmaZ=np.linspace(0, 2, k))
    file = open('data_shock_v_sigmaZ.dict', 'wb')
    pickle.dump(main(x), file)
    file.close()
    print("Completed in {}".format(dt.timedelta(seconds=time.time() - start)))

    """
    start = time.time()
    x = dict(sigmaZ=np.linspace(0, 2, k), ratio=np.linspace(0, 1, k))
    file = open('data_sigmaZ_v_ratio.dict', 'wb')
    pickle.dump(main(x), file)
    file.close()
    print("Completed in {}".format(dt.timedelta(seconds=time.time() - start)))
    """
