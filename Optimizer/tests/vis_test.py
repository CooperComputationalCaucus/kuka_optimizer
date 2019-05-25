''' Using the visualization function as a utility for exploring the operation steps in debugger'''

import sys
sys.path.append('../')
from bayes_opt import DiscreteBayesianOptimization, UtilityFunction
import numpy as np
from test_functions import PhilsFun
import matplotlib.pyplot as plt
from matplotlib import gridspec


def target(x):
    return PhilsFun().f(x)

_x = np.linspace(-2, 12.5, 10000).reshape(-1, 1)
_y = target(_x)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 12.5))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 12.5))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.show()
    
def step(dbo,sampler='KMBBO'):
    batch = dbo.suggest(utility, sampler = sampler, n_acqs = batch_size,  fit_gp = True, **kwargs)
    for point in batch:
        dbo.register(params=point,target=target(**point))
        #plot_gp(dbo,_x,_y)
        
if __name__=='__main__':
    # PARAMETERS
    prange = {'x':(-2,12.5,0.1)}
    random_state = 1234
    sampler = 'KMBBO'
    kwargs = {'multiprocessing':1}
    batch_size = 3
    
    KMBBO_steps=4
    greedy_steps=2
    
    # Initialize optimizer and utility function 
    dbo = DiscreteBayesianOptimization(f=None,
                                      prange=prange,
                                      random_state=random_state)
    utility = UtilityFunction(kind='ucb', kappa=5, xi=0.0)
    
    batch = [{'x':-2},{'x':12.5}]
    for point in batch:
        dbo.register(params=point,target=target(**point))
    #plot_gp(dbo,_x,_y)

    for _ in range(KMBBO_steps):
        step(dbo)
        
    for _ in range(greedy_steps):
        step(dbo,'greedy')