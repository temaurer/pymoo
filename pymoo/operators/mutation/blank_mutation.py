# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:30:32 2019

@author: temau
"""

import numpy as np

from pymoo.model.mutation import Mutation

class BlankMutation(Mutation):

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        return pop.new("X", X)
