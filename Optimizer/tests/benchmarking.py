from Optimizer.bayes_opt import BayesianOptimization, DiscreteBayesianOptimization
from Optimizer.bayes_opt.event import Events, DEFAULT_EVENTS
import time
import numpy as np
from Optimizer.bayes_opt import UtilityFunction

N_THREADS = 8


def run_function(func, prange, sampler, constraints=[], complements=False, verbose=True, batch_size=12, init_random=3,
                 n_batches=50, kappa=2.5, expected_max=None, **kwargs):
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
    if sampler == 'greedy': kwargs['complements'] = bool(complements)
    dbo = DiscreteBayesianOptimization(
        f=None,
        prange=prange,
        verbose=int(verbose),
        random_state=1,
        constraints=constraints
    )

    utility = UtilityFunction(kind='ucb', kappa=kappa, xi=0.0)
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
                pass
                # next_point[key] += np.random.uniform(-.001, .001)  # makes discrete point different than bin
            target = func(**next_point)
            dbo.register(params=next_point, target=target)
            if target > max_val['val']:
                max_val['val'] = target
                max_val['iter'] = idx
            if expected_max and target > 0.97 * expected_max and max_val['proximity'] == -1:
                max_val['proximity'] = idx
        if verbose: dbo.dispatch(Events.BATCH_END)
    if verbose: dbo.dispatch(Events.OPTMIZATION_END)
    end = time.time()
    print("Time taken for {} optimizaiton: {:8.2f} seconds".format(sampler, end - start))
    print("Maximum value {:.3f} found in {} batches".format(max_val['val'], max_val['iter'] + 1))
    return max_val


def run_parabolic(dim=4, constrained=True, verbose=True, batch_size=12, init_random=3, n_batches=25, strategy="all",
                  **kwargs):
    """
    Arguments
    =========
    dim: int, dimensionality of space and function
    constrained: boolean, will create constraints that hold sum below that required for max +eps.
    verbose: boolean, verbosity
    batch_size: int, number of points in a batch
    init_random: int, number of random batches
    n_batches: int, total number of batches to run before exit
    strategy: string, which or all search strategies
    """
    from Optimizer.tests.test_functions import Parabolic

    prange = {}
    pbounds = {}
    for i in range(dim):
        prange['x_{}'.format(i)] = (0, 1, 0.01)
        pbounds['x_{}'.format(i)] = (0, 1)
    parabolic = Parabolic('joint', dim)
    func = parabolic.f
    x = parabolic.arg_opt
    opt = func(**{'x_{}'.format(i): x[i] for i in range(len(x))})
    print("Actual optimum {:.3f} is at: {}".format(opt, x))
    constraints = []
    if constrained:
        constraints.append("{:.3f} - ".format(sum(x) * 1.1) + " - ".join(['x_{}'.format(i) for i in range(dim)]))
    print("Variable sum constraints are {} >=0".format(constraints))

    results = {}
    # Capitalist sampling parameters
    if strategy == "all" or strategy == "capitalist":
        capitalist_args = {'multiprocessing': N_THREADS,
                           'exp_mean': 1,
                           'n_splits': 4,
                           'n_iter': 250,
                           'n_warmup': 10000}
        capitalist_args.update(kwargs)
        results['capitalist'] = run_function(func,
                                             prange,
                                             sampler='capitalist',
                                             constraints=constraints,
                                             verbose=verbose,
                                             batch_size=batch_size,
                                             init_random=init_random,
                                             n_batches=n_batches,
                                             expected_max=float(dim),
                                             **capitalist_args
                                             )

    # Greedy sampling parameters
    if strategy == "all" or strategy == "greedy":
        greedy_args = {'multiprocessing': N_THREADS,
                       'n_iter': 250,
                       'n_warmup': 10000,
                       'kappa': 2.5}
        greedy_args.update(kwargs)
        results["greedy"] = run_function(func,
                                         prange,
                                         sampler='greedy',
                                         constraints=constraints,
                                         verbose=verbose,
                                         batch_size=batch_size,
                                         init_random=init_random,
                                         n_batches=n_batches,
                                         expected_max=float(dim),
                                         **greedy_args)

    # KMMBO sampling parameters
    if strategy == "all" or strategy == "KMBBO":
        KMBBO_args = {'multiprocessing': N_THREADS,
                      'n_slice': 500}
        KMBBO_args.update(kwargs)
        results['KMBBO'] = run_function(func,
                                        prange,
                                        sampler='KMBBO',
                                        constraints=constraints,
                                        verbose=verbose,
                                        batch_size=batch_size,
                                        init_random=init_random,
                                        n_batches=n_batches,
                                        expected_max=float(dim),
                                        **KMBBO_args)
    return results


def loop_parabolic(dims=[3, 4, 5], verbose=True, batch_size=12, init_random=3, n_batches=25, strategy="all",
                   strat_args=[]):
    """
    Arguments
    =========
    dims: list of ints, set of parabolic dimensionalities to explore
    constrained: boolean, will create constraints that hold sum below that required for max +eps.
    verbose: boolean, verbosity
    batch_size: int, number of points in a batch
    init_random: int, number of random batches
    n_batches: int, total number of batches to run before exit
    strategy: string, which or all search strategies
    strat_args: list of strategy kwargs dicts to cycle
    """
    if strategy == "all":
        for dim in dims:
            print("Optimizing parabolic function with {} dimensions".format(dim))
            run_parabolic(dim=dim,
                          verbose=verbose,
                          batch_size=batch_size,
                          init_random=init_random,
                          n_batches=n_batches)
            print("".join(["=" for _ in range(80)]))
            print()
            print()
    else:
        results = {}
        for kwargs in strat_args:
            print(kwargs)
            results[str(kwargs)] = {}
            for i in range(1):
                res = run_parabolic(verbose=verbose,
                                    n_batches=n_batches,
                                    strategy=strategy,
                                    **kwargs)
                results[str(kwargs)][i] = res[strategy]
        return results


def run_pH_model(verbose=True, batch_size=12, init_random=3, n_batches=25):
    """
    Arguments
    =========
    dim: int, dimensionality of space and function
    constrained: boolean, will create constraints that hold sum below that required for max +eps.
    verbose: boolean, verbosity
    batch_size: int, number of points in a batch
    init_random: int, number of random batches
    n_batches: int, total number of batches to run before exit
    """
    from Optimizer.tests.test_functions import HERVirtualModel

    complements = True
    model = HERVirtualModel()
    func = model.f
    constraints = model.constraints
    prange = model.prange

    # Capitalist sampling parameters
    sampler = "capitalist"
    capitalist_args = {'multiprocessing': N_THREADS,
                       'exp_mean': 1,
                       'n_splits': 4,
                       'n_iter': 250,
                       'n_warmup': 10000}
    max_val = run_function(func,
                           prange,
                           sampler='capitalist',
                           constraints=constraints,
                           verbose=verbose,
                           batch_size=batch_size,
                           init_random=init_random,
                           n_batches=n_batches,
                           **capitalist_args)

    # Greedy sampling parameters
    greedy_args = {'multiprocessing': N_THREADS,
                   'n_iter': 1000,
                   'n_warmup': 250}
    max_val = run_function(func,
                           prange,
                           sampler='greedy',
                           constraints=constraints,
                           complements=complements,
                           verbose=verbose,
                           batch_size=batch_size,
                           init_random=init_random,
                           n_batches=n_batches,
                           **greedy_args)

    # KMMBO sampling parameters
    KMBBO_args = {'multiprocessing': N_THREADS,
                  'n_slice': 500}
    max_val = run_function(func,
                           prange,
                           sampler='KMBBO',
                           constraints=constraints,
                           verbose=verbose,
                           batch_size=batch_size,
                           init_random=init_random,
                           n_batches=n_batches,
                           **KMBBO_args)

def greedy_main():
    strat_args = [{'dim': 5, 'batch_size': 16, 'init_random': 3, 'kappa': 1},
                  {'dim': 5, 'batch_size': 16, 'init_random': 3, 'kappa': 2.5},
                  {'dim': 5, 'batch_size': 16, 'init_random': 3, 'kappa': 5},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'kappa': 1},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'kappa': 2.5},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'kappa': 5},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'kappa': 1},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'kappa': 2.5},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'kappa': 5},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'kappa': 1},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'kappa': 2.5},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'kappa': 5}
                  ]
    strat_args = [{'dim': 8, 'batch_size': 48, 'init_random': 1, 'kappa': 1},
                  {'dim': 8, 'batch_size': 48, 'init_random': 1, 'kappa': 2.5},
                  {'dim': 8, 'batch_size': 48, 'init_random': 1, 'kappa': 5}
                  ]
    res = loop_parabolic(strategy='greedy', strat_args=strat_args, n_batches=20)
    with open("greedy_benchmarking.csv", 'w') as f:
        for args in res:
            f.write(args + "\n")
            f.write("n,iter_near,iter_found,max_val_found\n")
            f.write("".join(['=' for _ in range(80)]) + "\n")
            for i in res[args]:
                f.write("{},{},{},{}\n".format(i, res[args][i]['proximity'], res[args][i]['iter'], res[args][i]['val']))
            f.write("\n")

def capitalism_main():
    strat_args = [{'dim': 5, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 1},
                  {'dim': 5, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 2.5},
                  {'dim': 5, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 0.5},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 1},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 2.5},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 0.5},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 1},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 2.5},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 4, 'init_random': 3, 'exp_mean': 0.5},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 1},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 2.5},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 4, 'init_random': 1, 'exp_mean': 0.5},
                  {'dim': 5, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 1},
                  {'dim': 5, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 2.5},
                  {'dim': 5, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 0.5},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 1},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 2.5},
                  {'dim': 5, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 0.5},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 1},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 2.5},
                  {'dim': 10, 'batch_size': 16, 'n_splits': 8, 'init_random': 3, 'exp_mean': 0.5},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 1},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 2.5},
                  {'dim': 10, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 0.5}
                  ]
    strat_args = [{'dim': 8, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 1},
                  {'dim': 8, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 2.5},
                  {'dim': 8, 'batch_size': 48, 'n_splits': 8, 'init_random': 1, 'exp_mean': 0.5}
                  ]
    res = loop_parabolic(strategy='capitalist', strat_args=strat_args, n_batches=20)
    with open("capitalism_benchmarking.csv", 'w') as f:
        for args in res:
            f.write(args + "\n")
            f.write("n,iter_near,iter_found,max_val_found\n")
            f.write("".join(['=' for _ in range(80)]) + "\n")
            for i in res[args]:
                f.write("{},{},{},{}\n".format(i, res[args][i]['proximity'], res[args][i]['iter'], res[args][i]['val']))
            f.write("\n")


def KMBBO_main():
    strat_args = [{'dim': 5, 'batch_size': 16, 'init_random': 3, 'n_slice': 250},
                  {'dim': 5, 'batch_size': 16, 'init_random': 3, 'n_slice': 500},
                  {'dim': 5, 'batch_size': 16, 'init_random': 3, 'n_slice': 1000},
                  {'dim': 5, 'batch_size': 16, 'init_random': 3, 'n_slice': 5000},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'n_slice': 250},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'n_slice': 500},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'n_slice': 1000},
                  {'dim': 5, 'batch_size': 48, 'init_random': 1, 'n_slice': 5000},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'n_slice': 250},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'n_slice': 500},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'n_slice': 1000},
                  {'dim': 10, 'batch_size': 16, 'init_random': 3, 'n_slice': 5000},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'n_slice': 250},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'n_slice': 500},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'n_slice': 1000},
                  {'dim': 10, 'batch_size': 48, 'init_random': 1, 'n_slice': 5000}
                  ]
    strat_args = [{'dim': 8, 'batch_size': 48, 'init_random': 1, 'n_slice': 250},
                  {'dim': 8, 'batch_size': 48, 'init_random': 1, 'n_slice': 500},
                  {'dim': 8, 'batch_size': 48, 'init_random': 1, 'n_slice': 1000},
                  {'dim': 8, 'batch_size': 48, 'init_random': 1, 'n_slice': 5000}
                  ]
    res = loop_parabolic(strategy='KMBBO', strat_args=strat_args, n_batches=20)
    with open("KMBBO_benchmarking.csv", 'w') as f:
        for args in res:
            f.write(args + "\n")
            f.write("n,iter_near,iter_found,max_val_found\n")
            f.write("".join(['=' for _ in range(80)]) + "\n")
            for i in res[args]:
                f.write("{},{},{},{}\n".format(i, res[args][i]['proximity'], res[args][i]['iter'], res[args][i]['val']))
            f.write("\n")
if __name__ == '__main__':
    capitalism_main()
    greedy_main()
    KMBBO_main()
    # loop_parabolic([i for i in range(9,10)], constrained=True, n_batches=15)
    # loop_parabolic([i for i in range(3,5)], constrained=False, n_batches=25)
    # run_pH_model(n_batches=15)
