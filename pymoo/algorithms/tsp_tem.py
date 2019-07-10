# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:49 2019

@author: temau
"""
import numpy as np
import autograd.numpy as anp
from pymop.problem import Problem
from pymoo.algorithms.unsga3 import unsga3
from pymoo.optimize import minimize

class Route():
    #need cities in order PLUS total distance
    def __init__(self, sol_temp, dist):
        self.sol_temp = sol_temp
        self.dist = dist


def _City(cities):    
    city_num = int(cities.size/2)
    #print(city_num)
    dist_Mat = np.zeros((city_num,city_num))
    for i in range(city_num):
        for j in range(city_num):
            p_1 = cities[i]
            p_2 = cities[j]
            dist = np.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)
            dist_Mat[i][j] = dist
    return dist_Mat

#def _City(cities):
#    def __init__(self, cities):    
#        city_num = int(cities.size/2)
#        #print(city_num)
#        dist_Mat = np.zeros((city_num,city_num))
#        for i in range(city_num):
#            for j in range(city_num):
#                p_1 = cities[i]
#                p_2 = cities[j]
#                dist = np.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)
#                dist_Mat[i][j] = dist
#        return dist_Mat
#  
    
#    index = np.array(range(0,city_num))
#    shuffle(index)
#    sol_temp = np.zeros((city_num+1,2))
#    #print(cities.size)
#    p_1 = cities[index[0]]
#    #print(index[0])
    #print(p_1)
    #print(p_1[0])
    #print(p_1[1])
#    d_tot = 0
#    for i in range(1,city_num):
#        p_2 = cities[index[i]]
#        d_tot += np.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)
#        sol_temp[i-1] = p_1
#        sol_temp[i] = p_2
#        p_1 = p_2
#        #print(sol_temp[i-1])
#        #print(sol_temp[i])
#    p_1 = cities[index[0]]
#    sol_temp[city_num] = p_1
#    d_tot += np.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)
#    elem = Route(sol_temp,d_tot)
#    print(elem.sol_temp)
#    print(elem.dist)
#    return elem

cities = np.array([[3., 4.],[2.,5.],[4.,5.],[2.,3.],[6.,10.]])
dist_Mat = _City(cities)

def _getDist(dist_Mat, index):
    #whole population for one generation passed in
    city_order = np.argsort(index)
    city_num = len(dist_Mat)
    d_tot = 0
    #print("city_order shape: ")
    #print(city_order.shape)
    #print(city_order)
    (r,c) = city_order.shape
    #f is 2-D: population number, number of objectives
    f = np.zeros((10,2))
    #print(np.size(f))
    for j in range(r):
        for i in range(city_num):
            d_tot += dist_Mat[(city_order[j][i])][city_order[j][np.mod(i+1,city_num)]]
        #add another objective that is constant value
        #print("This is d_tot: ")
        #print(d_tot)
        f[j] =[d_tot, d_tot]
    #print("This is f: ")
    #print(f)
#   sum(anp.power(x, 2) - self.const_1 * anp.cos(2 * anp.pi * x), axis=1) 
    #return d_tot
    #out["F"] = anp.sum(d_tot, axis=0)
    return f

# always derive from the main problem for the evaluation
class MyProblem(Problem):
    def __init__(self, const_1=5, const_2=0.1):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = 0 * anp.ones(5)
        xu = 1 * anp.ones(5)
        #set the number of variables, number of objectives, number of constraints, lower and upper limit, respectively
        super().__init__(n_var=5, n_obj=2, n_constr=0, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
        #self.const_1 = const_1
        #self.const_2 = const_2
        
    # implemented the function evaluation function - the arrays to fill are provided directly
    #note that index is is entire population for one generation
    def _evaluate(self,index, out, *args, **kwargs):
        # an objective function to be evaluated using set of city vars
        f = _getDist(dist_Mat,index)
        #f is single value of optimized function (because single objective) for each population member, so a 1D array the size of population output for F
        #not quite sure yet what exactly f is supposed to be when multi-objective
        out["F"] = f
        #G is constraints for entire generation (size should be (pop_size, const_num))
        #out["G"] = np.zeros((10,1))
        # !!! only if a constraint value is positive it is violated !!!
        # set the constraint that x1 + x2 > var2

        # set the constraint that x3 + x4 < var2
        #g2 = self.const_2 - (x[:, 2] + x[:, 3])
        #g1 = (x[:, 0] + x[:, 1]) - self.const_2

           
    

problem = MyProblem()
F, G, CV, feasible, dF, dG = problem.evaluate(anp.random.rand(10, 5), 
                                              return_values_of=["F", "G", "CV", "feasible", "dF", "dG"])
# get the optimal solution of the problem for the purpose of comparison
pf = problem.pareto_front()

# create the algorithm object
method = unsga3(ref_dirs=np.array([[15.,13.]]), pop_size=10, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
                  method,
                  termination=('n_gen', 20),
                  pf=pf,
                  disp=True)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)



