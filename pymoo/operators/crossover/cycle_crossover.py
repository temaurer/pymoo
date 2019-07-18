# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:11:49 2019

@author: temau
"""

import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
from pymoo.rand import random


class CycleCrossover(Crossover):
    
    def __init__(self, prob, var_type=np.int):
        super().__init__(2, 2)
        self.prob = float(prob)
        self.var_type = var_type

    def _do(self, problem, pop, parents, **kwargs):
        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        _, n_matings, n_var = X.shape
        print("Number of matings: %s" %n_matings)
        #print("Number of variables: %s" %n_var)
        #print("Number of offspring: %s" %self.n_offsprings)
        children = np.full((self.n_offsprings*n_matings, problem.n_var), np.inf)
        k=0
        while k < n_matings*self.n_offsprings:
            x1 = []
            x2 = []
            #not sure about this method for selecting x1 and x2... need to preserve elitism more
            #while x1==x2:
            try:
                x1 = X[0][random.randint(0,n_matings-1)].copy()
                x1 = x1.tolist()
                x2 = X[1][random.randint(0,n_matings-1)].copy()
                x2 = x2.tolist()
            except:
                x1 = X[0][0].copy()
                x1 = x1.tolist()
                x2 = X[0][0].copy()
                x2 = x2.tolist()
            #print("X1:")
            #print(x1)
            #print("X2:")
            #print(x2)
            og_x1 = x1.copy()
            og_x2 = x2.copy()
            cycles = []
            cycles_index1 = []
            cycles_index2 = []
            num_cyc = 0
            while len(x1) > 1:
                cycle = []
                cycle_index1 = []
                cycle_index2 = []
                index=0
                cycle.append(x1[index])
                index_1 = og_x1.index(x1[index])
                cycle_index1.append(index_1)
                index_2 = og_x2.index(x1[index])
                cycle_index2.append(index_2)
                x1.pop(index)
                while cycle[0] != x2[index]:
                    temp = x2[index]
                    x2.pop(index)
                    cycle.append(temp)
                    #note: using obj.index may be slow for long lists
                    index = x1.index(temp)
                    index_1 = og_x1.index(temp)
                    cycle_index1.append(index_1)
                    x1.pop(index)
                    index_2 = og_x2.index(temp)
                    cycle_index2.append(index_2)
                num_cyc = num_cyc + 1
                x2.pop(index)
                cycles_index1.append(cycle_index1)
                cycles_index2.append(cycle_index2)
                cycles.append(cycle)
            if len(x1) == 1:
                cycles.append([x2[0]])
                x1.pop
                x2.pop
                index_2=og_x2.index(x2[0])
                index_1=og_x1.index(x1[0])
                cycles_index2.append([index_2])
                cycles_index1.append([index_1])
                num_cyc = num_cyc + 1
            child1 = np.zeros(len(og_x1))
            child2 = np.zeros(len(og_x2))
            i=1
            #print("Num cyc %s" %num_cyc)
            while i<num_cyc:
                child1[cycles_index1[i-1]] = cycles[i-1]
                child2[cycles_index2[i-1]] = cycles[i-1]
                child1[cycles_index2[i]] = cycles[i]
                child2[cycles_index1[i]] = cycles[i]
                i +=2
            if np.mod(num_cyc,2)!=0:
                child1[cycles_index1[i-1]] = cycles[i-1]
                child2[cycles_index2[i-1]] = cycles[i-1]
            children[k] = child1
            try:
                children[k+1] = child2
            except:
                pass
            k += 2
            #The method to fill children[] is fine for now as long as k's limit is even
        #print("Children")
        children = children.astype(int)
        #print(children)
        return pop.new("X", children)
