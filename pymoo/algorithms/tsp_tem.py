# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:49 2019

@author: temau
"""
import numpy as np
import autograd.numpy as anp
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.unsga3 import unsga3

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

def _getDist(dist_Mat, index, out):
    city_order = np.argsort(index)
    city_num = len(dist_Mat)
    d_tot = 0
    for i in range(city_num):
        d_tot += dist_Mat[(city_order[i])][city_order[np.mod(i+1,city_num)]]
    return d_tot
    #out["F"] = anp.sum(d_tot, axis=0)

# always derive from the main problem for the evaluation
class MyProblem(Problem):
    def __init__(self, const_1=5, const_2=0.1):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = 0 * anp.ones(5)
        xu = 1 * anp.ones(5)

        super().__init__(n_var=5, n_obj=1, n_constr=2, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
        #self.const_1 = const_1
        #self.const_2 = const_2
        
    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self,index, out, *args, **kwargs):
        # an objective function to be evaluated using set of city vars
        f = _getDist(dist_Mat,index,out)
        out["F"] = f
        out["G"] = np.array([[0],[1]])
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
method = unsga3(ref_dirs=np.array([[15.]]), pop_size=10, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
                  method,
                  termination=('n_gen', 20),
                  pf=pf,
                  disp=True)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)



