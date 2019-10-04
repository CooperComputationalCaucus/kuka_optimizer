from Optimizer.bayes_opt import DiscreteBayesianOptimization
from Optimizer.bayes_opt.event import Events
from Optimizer import bayes_opt
import time
import numpy as np
from Optimizer.bayes_opt import UtilityFunction
from Optimizer.tests.test_functions import GPVirtualModel
import pandas as pd
import os, shutil

N_THREADS = 7
CAPITALIST_ARGS = {'multiprocessing': N_THREADS,
                   'exp_mean': 2.5,
                   'n_splits': 14,
                   'n_iter': 250,
                   'n_warmup': 1000
                   }


def run_function(func, prange, constraints=[], verbose=True, batch_size=12, init_random=3,
                 n_batches=50, expected_max=None, **kwargs):
    """
    Arguments
    =========
    func: function for dimensionality of prange
    prange: list of 3-tuples for variable min, max, step
    sampler: string, greedy or KMBBO
    verbose: boolean, verbosity
    batch_size: int, number of points in a batch
    init_random: int, number of random batches
    kwargs: for greedy or KMBBO sampler
    expected_max: float, expected maximum for proximity return

    Returns
    ======
    max_val: dict, of iteration when within expected max, when current max is found, and current max
    """
    start = time.time()
    sampler = "capitalist"
    dbo = DiscreteBayesianOptimization(
        f=None,
        prange=prange,
        verbose=int(verbose),
        random_state=1,
        constraints=constraints
    )

    utility = None
    max_val = {'proximity': -1, 'iter': -1, 'val': 0}
    if verbose:
        dbo._prime_subscriptions()
        dbo.dispatch(Events.OPTMIZATION_START)
    for idx in range(n_batches):
        if idx < init_random:
            if dbo.constraints:
                next_points = [dbo._space.array_to_params(x) for x in dbo.constrained_rng(batch_size, bin=True)]
            else:
                next_points = [dbo._space.array_to_params(dbo.space._bin(
                    dbo._space.random_sample(constraints=dbo.get_constraint_dict()))) for _ in range(batch_size)]

        else:
            next_points = dbo.suggest(utility,
                                      sampler=sampler,
                                      n_acqs=batch_size,
                                      **kwargs)
        for next_point in next_points:
            for key in next_point:
                next_point[key] += np.random.uniform(-.01, .01)  # makes discrete point different than bin
                if next_point[key] < 0: next_point[key] = 0.
            target = func(**next_point)
            if target < 0:
                target = 0
            dbo.register(params=next_point, target=target)
            if target > max_val['val']:
                max_val['val'] = target
                max_val['iter'] = idx
            if expected_max and target > 0.97 * expected_max and max_val['proximity'] == -1:
                max_val['proximity'] = idx
        if verbose:
            dbo.dispatch(Events.BATCH_END)
    if verbose:
        dbo.dispatch(Events.OPTMIZATION_END)
    end = time.time()
    print("Time taken for {} optimizaiton: {:8.2f} seconds".format(sampler, end - start))
    print("Maximum value {:.3f} found in {} batches".format(max_val['val'], max_val['iter'] + 1))
    return dbo


def run_GP_model(prange, path, constraints=[], noisy=False, verbose=True, batch_size=12, init_random=3,
                 n_batches=25):
    virtual_model = GPVirtualModel(path, noisy=noisy)
    pbounds = {}
    for key, value in prange.items():
        pbounds[key] = (value[0], value[1])
    func = virtual_model.f
    print("Experimental optimum {:.3f} is at: {}".format(virtual_model.exp_max, virtual_model.param_max))
    dbo = run_function(func,
                       prange,
                       constraints=constraints,
                       verbose=verbose,
                       batch_size=batch_size,
                       init_random=init_random,
                       n_batches=n_batches,
                       expected_max=virtual_model.exp_max,
                       **CAPITALIST_ARGS
                       )

    return dbo


def loop_GP_model(n_iter, prange, path, constraints=[], noisy=False, verbose=True, batch_size=12, init_random=3,
                  n_batches=25, outdir='./tmp'):
    os.makedirs(outdir, exist_ok=True)
    for i in range(n_iter):
        _path = os.path.join(outdir, '{}.csv'.format(i))
        dbo = run_GP_model(prange,
                           path,
                           constraints=constraints,
                           noisy=noisy,
                           verbose=verbose,
                           batch_size=batch_size,
                           init_random=init_random,
                           n_batches=n_batches)
        gp = dbo._gp
        df = pd.concat(
            [pd.DataFrame(gp.X_train_, columns=prange.keys()), pd.DataFrame(gp.y_train_, columns=['Target'])],
            axis=1)
        df.to_csv(_path)


def gp_main(outdir='./tmp'):
    n_iter = 50
    prange = {'AcidRed871_0gL': (0, 5, .25),
              'L-Cysteine-50gL': (0, 5, .25),
              'MethyleneB_250mgL': (0, 5, .25),
              'NaCl-3M': (0, 5, .25),
              'NaOH-1M': (0, 5, .25),
              'P10-MIX1': (1, 5, 0.2,),
              'PVP-1wt': (0, 5, .25),
              'RhodamineB1_0gL': (0, 5, .25),
              'SDS-1wt': (0, 5, .25),
              'Sodiumsilicate-1wt': (0, 5, .25)}
    path = 'GPVirtualModel.pkl'
    constraints = [
        '5 - L-Cysteine-50gL - NaCl-3M - NaOH-1M - PVP-1wt - SDS-1wt - Sodiumsilicate-1wt - AcidRed871_0gL - '
        'RhodamineB1_0gL - MethyleneB_250mgL']
    loop_GP_model(n_iter,
                  prange,
                  path=path,
                  constraints=constraints,
                  noisy=True,
                  verbose=1,
                  batch_size=14,
                  init_random=2,
                  n_batches=25,
                  outdir=outdir)


if __name__ == "__main__":
    import sys
    import multiprocessing
    sys.path.append('../')

    N_PROCS = 1
    procs=[]
    for p in range(N_PROCS):
        outdir = './tmp_{}'.format(p)
        proc = multiprocessing.Process(target=gp_main, args=(outdir,))
        procs.append(proc)
        proc.start()
