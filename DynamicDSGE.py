"""
Class container for the Dynamic DSGE model
"""

__author__ = "Karl Naumann"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import pandas as pd


class DynamicDSGE(object):
    def __init__(self):
        pass

    def set_params(self):
        pass

    def graph(self):
        pass

    def simulate(self, t_end: int, start: np.ndarray) -> pd.DataFrame:
        pass

    def  _step(self, x: np.ndarray, t:float) -> np.ndarray:
        pass



if __name__ == "__main__":
    pass
