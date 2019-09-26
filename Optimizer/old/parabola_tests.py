'''
Optimisation runs of BayesianOptimization on very simple functions: joint parabola (actually, 2-norm) 
and independent parabolas with different sensitivities. They form an easy case. The purpose is to test
performance of the library.
'''

from bayes_opt import BayesianOptimization
import numpy as np
np.set_printoptions(precision=3)
import time


DIM = 3 #14
print(f'The number of parameters is set to {DIM}')
arg_opt = np.random.random_sample((DIM,))
arg_opt = np.around(arg_opt, decimals=3)
print(f'The optimum is at {arg_opt}')

sensitivity = np.random.random_sample((DIM,))
sensitivity = np.around(sensitivity*10, decimals=3)
print(f'The influence of different coordinates on the value of \"independent\" parabola is {sensitivity}')


init_points = 2 #2000
n_iter = 1  #1
verbose = 2
print(f'Each optimisation will start with {init_points} inital points and do {n_iter} steps using GPs')


pbounds = {}

for i in range(DIM):
    pbounds[f'x_{i}'] = (0,1)

def parabola_joint(**kwargs):
    global DIM, arg_opt
    point = np.zeros(DIM)
    for i in range(DIM):
        point[i] = kwargs[f'x_{i}']

    return DIM - 5*np.linalg.norm(point - arg_opt)

def parabola_independent(**kwargs):
    global DIM, arg_opt
    point = np.zeros(DIM)
    for i in range(DIM):
        point[i] = kwargs[f'x_{i}']

    return DIM - sum([sensitivity[i]*(point[i]-arg_opt[i])**2 for i in range(DIM)])

print("Optimising the joint parabola")

optimizer = BayesianOptimization(
    f=parabola_joint,
    pbounds=pbounds,
    verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    # random_state=1,
)

start = time.time()
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter,
)
end = time.time()
print('Maximum value found using GPs: ', optimizer.max['target'])
print(f'The optimum: {DIM}')
print(f'It took {int(end - start)} seconds to do {n_iter} iterations')

print("Optimising the independent parabolas")

optimizer = BayesianOptimization(
    f=parabola_independent,
    pbounds=pbounds,
    verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    # random_state=1,
)

start = time.time()
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter,
)
end = time.time()

print('Maximum value found using GPs: ', optimizer.max['target'])
print(f'The optimum: {DIM}')
print(f'It took {int(end - start)} seconds to do {n_iter} iterations')
