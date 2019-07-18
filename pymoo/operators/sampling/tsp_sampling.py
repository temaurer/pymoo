# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:16:45 2019

@author: temau
"""

from pymoo.model.sampling import Sampling
from pymoo.rand import random
import numpy as np


class TSPSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def sample(self, problem, pop, n_samples, **kwargs):
        m = problem.n_var
#        val = random.random(size=(n_samples, m))
        parent0 = np.zeros((n_samples,m))
        for i in range(n_samples):
            my_list = list(range(m))
            my_array = np.array(my_list)
            my_array = random.shuffle(my_array)
#            my_array = np.array(my_list)
            parent0[i] = my_array
        parent0 = parent0.astype(int)
        return pop.new("X", parent0)

