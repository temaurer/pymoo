import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.nsga3 import nsga3
from pymoo.util import plotting
from pymop.factory import get_problem

# load a test or define your own problem
problem = get_problem("zdt1")

# get the optimal solution of the problem for the purpose of comparison
pf = problem.pareto_front()

# create the algorithm object
method = nsga3(ref_dirs=np.array([[3., 4.],[2.,5],[4.,5],[2.,5]]), pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
                  method,
                  termination=('n_gen', 200),
                  pf=pf,
                  disp=True)

# plot the results as a scatter plot
plotting.plot(pf, res.F, labels=["Pareto-Front", "F"])