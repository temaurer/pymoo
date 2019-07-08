import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.unsga3 import unsga3
from.pymoo.algorithms.ga import ga
from pymoo.util import plotting
from pymop.factory import get_problem

# load a test or define your own problem
problem = get_problem("g01")

# get the optimal solution of the problem for the purpose of comparison
#pf = problem.pareto_front()

# create the algorithm object
#method = unsga3(ref_dirs=np.array([[3.]]), pop_size=100, elimate_duplicates=True)

method = ga(pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
                  method,
                  termination=('n_gen', 200),
                  #pf=pf,
                  disp=True)

# plot the results as a scatter plot
#plotting.plot(pf, res.F, labels=["Pareto-Front", "F"])