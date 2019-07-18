# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:49 2019

@author: temau
"""
import numpy as np
import autograd.numpy as anp
from pymop.problem import Problem
from pymoo.algorithms.unsga3 import unsga3
#from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy import random
from pymoo.operators.crossover.tem_crossover import CycleCrossover
from pymoo.operators.sampling.tem_sampling import TSPSampling
from pymoo.operators.mutation.blank_mutation import BlankMutation

#In nsga3, the default values for crossover and mutation were changed
#Changed:
#   SimulatedBinaryCrossover(prob=0.8, eta=30)
#   PolynomialMutation(prob=0.1, eta=20)

#Original:
#   SimulatedBinaryCrossover(prob=1.0, eta=30))
#   PolynomialMutation(prob=None, eta=20))

startTime = datetime.now()


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
    #print(dist_Mat)
    return dist_Mat

#shanghai,beijing,hong kong, singapore, xiamen, tokyo, bangkok, ho chi minh city
#cities = np.array([[31.2304, 121.4737],[39.9042,116.4074],[22.3193,114.1694],[1.3521,103.8198],[24.45964,118.08954],[35.68321,139.80894],[13.75335,100.50483],[10.77824,106.70324]])
cities = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556], [3238, 1229], [4196, 1004], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367], [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826], [2370, 2975]])

dist_Mat = _City(cities)

def _getDist(dist_Mat, index):                   
    (r,c) = index.shape
    f = np.zeros((r,c))
    d_tot = 0
    city_num = c
    print("made it to get dist")
    for j in range(r):             
        for i in range(city_num):
         #   print("Right before d_tot calculation")
            #if index[j][i] < 3:
          #      print("so that worked")
           #     print(index[j][i])
            d_tot += int(dist_Mat[int(index[j][i])][int(index[j][np.mod(i+1,city_num)])])
           # print("Ok so I guess it calculated something")
        f[j] = int(d_tot)
    return f

# always derive from the main problem for the evaluation
class MyProblem(Problem):
    def __init__(self, const_1=5, const_2=0.1):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = 0 * anp.ones(31)
        xu = 1 * anp.ones(31)
        #set the number of variables, number of objectives, number of constraints, lower and upper limit, respectively
        super().__init__(n_var=31, n_obj=1, n_constr=0, xl=xl, xu=xu, evaluation_of="auto", type_var = np.int)

        
    # implemented the function evaluation function - the arrays to fill are provided directly
    #note that index is is entire population for one generation
    def _evaluate(self,index, out, *args, **kwargs):
        # an objective function to be evaluated using set of city vars
        print("This is index")
        print(index)
        f = _getDist(dist_Mat,index)
        print("made it past the getDist")
        #f is single value of optimized function (because single objective) for each population member, so a 1D array the size of population output for F
        #not quite sure yet what exactly f is supposed to be when multi-objective
        out["F"] = f
        
           
    
best_of_gen = []
problem = MyProblem()
#parent0 = randint(31,size=(10,31))
#for i in range(10):
#    my_list = list(range(31))
#    random.shuffle(my_list)
#    parent0[i] = my_list
##anp.random.rand(100,8) initializes first generation (parents) as 100 arrays of 8 vals 0-1
#F, G, CV, feasible, dF, dG = problem.evaluate(parent0, 
#                                              return_values_of=["F", "G", "CV", "feasible", "dF", "dG"])
## get the optimal solution of the problem for the purpose of comparison
#pf = problem.pareto_front()

# create the algorithm object
method = unsga3(ref_dirs=np.array([[15000.]]), pop_size=10, sampling= TSPSampling(10),#get_sampling("int_random"), #TSPSampling(30),
                       crossover=CycleCrossover(prob=0.5),
                       mutation=BlankMutation(prob=0.5), elimate_duplicates=True)
#method = get_algorithm("unsga3", ref_dirs=np.array([[15000.]]), pop_size=800, sampling=get_sampling("int_random"), crossover=get_crossover("int_sbx", prob=1.0, eta=3.0), mutation=get_mutation("int_pm", eta=3.0), elimate_duplicates=True)

#note: the parameters of population and generation influence the performance greatly
#The parameters of probability of mutation and crossover were not found to be significant
# execute the optimization
res = minimize(problem,
                  method,
                  termination=('n_gen', 4),
#                  pf=pf,
                  disp=True)
                  
comput_time = datetime.now() - startTime
print("The time for the program to terminate: %s" %comput_time)
var_real = res.X.astype(np.float)
print("This is var_real")
print(var_real)
var_order = np.zeros((1,31))
c = 31
var_order = var_order.astype(int)
print("Best solution found: %s" % var_order)
print("Function value: %s" % res.F)
#var = res.X.astype(np.int))
#print("City order: %s" %np.argsort(var))
city_graph_x = []
city_graph_y = []
for i in range(len(var_order)):
    city_graph_x.append(cities[i][0])
    city_graph_y.append(cities[i][1])
city_graph_x.append(cities[0][0])
city_graph_y.append(cities[0][1])
f1 = plt.figure(1)
plt.plot(city_graph_x, city_graph_y, color='green', linestyle='dashed', linewidth = 3, marker = 'o', markerfacecolor = 'blue', markersize = 12)

for i in range(len(var_order)):
    plt.annotate(str(var_order[i]),(city_graph_x[i],city_graph_y[i]),fontsize=20)
f2 = plt.figure(2)
plt.plot(range(11),best_of_gen, color = 'grey', linestyle = 'solid', linewidth = 0.5)
plt.show()
print(best_of_gen)

