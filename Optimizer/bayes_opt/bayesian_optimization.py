import warnings
import numpy as np
from multiprocessing import Pool

from .target_space import TargetSpace, DiscreteSpace, PartnerSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger, _get_discrete_logger
from .util import UtilityFunction, acq_max, ensure_rng,get_rnd_quantities
from .parallel_opt import disc_acq_max, disc_acq_KMBBO
from .parallel_opt import disc_constrained_acq_max, disc_constrained_acq_KMBBO

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import pandas as pd
import re


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback == None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2, constraints=[]):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state,
        )

        self._verbose = verbose
        # Key constraints correspond to literal keyword names
        # array constraints correspond to point in array row
        self._key_constraints = constraints
        self._array_constraints = self.array_like_constraints()
        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    @property
    def constraints(self):
        return self._array_constraints
    
    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if isinstance(params,list):
            for param in params:
                if lazy:
                    self._queue.add(param)
                else:
                    self._space.probe(param)
                    self.dispatch(Events.OPTMIZATION_STEP)
        else:
            if lazy:
                self._queue.add(params)
            else:
                self._space.probe(params)
                self.dispatch(Events.OPTMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTMIZATION_START, _logger)
            self.subscribe(Events.OPTMIZATION_STEP, _logger)
            self.subscribe(Events.OPTMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

        self.dispatch(Events.OPTMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
        
    def array_like_constraints(self):
        '''
        Takes list of logical constraints in terms of space keys,
        and replaces the constraints in terms of array indicies. 
        This allows direct evaluation in the acquisition function.
        Parameters
        ----------
        constraints: list of string constraints
        '''
        keys = self.space.keys
        array_like = []
        for constraint in self._key_constraints: 
             tmp = constraint
             for idx,key in enumerate(keys):
                  #tmp = tmp.replace(key,'x[0][{}]'.format(idx))
                  tmp = tmp.replace(key,'x[{}]'.format(idx))
             array_like.append(tmp)
        return array_like
    
    def get_constraint_dict(self):
        '''
        Develops inequality constraints ONLY. (>=0)
        '''
        #TODO: write function to return scipy constraint dictionary for optimizer
        #TODO: write options for equality constraints (incorporate in randomizer)
        #TODO: write options for jacobian if needed?
        dicts = []
        funcs=[]
        for idx,constraint in enumerate(self.constraints):
            st = "def f_{}(x): return pd.eval({})\nfuncs.append(f_{})".format(idx,constraint,idx)
            exec(st)
            dicts.append({'type': 'ineq',
                        'fun':funcs[idx]})
        return dicts
    
    
class DiscreteBayesianOptimization(BayesianOptimization):
    '''
    Optimization object by default performs batch optimization of discrete parameters. 
    When using the open form optimizer (i.e. writing loops manually) the suggested parameters handled as lists of dicts. 
    
    '''
    def __init__(self, f, prange, random_state=None, verbose=2,constraints=[]):
        """"""
        
        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._pbounds = {item[0] :(item[1][:2]) for item in sorted(prange.items(), key=lambda x: x[0])}   
        super(DiscreteBayesianOptimization, self).__init__(f,self._pbounds,random_state,verbose,constraints)
        self._space = DiscreteSpace(f, prange, random_state)
        self.partner_space = PartnerSpace(f,prange,random_state)
        
    def probe(self, params, lazy=True):
        """Probe target of x"""
        if isinstance(params,list):
            for param in params:
                if lazy:
                    self._queue.add(param)
                else:
                    self._space.probe(param)
                    self.dispatch(Events.OPTMIZATION_STEP)
        else:
            if lazy:
                self._queue.add(params)
            else:
                self._space.probe(params)
                self.dispatch(Events.OPTMIZATION_STEP)     
    
    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_discrete_logger(self._verbose)
            self.subscribe(Events.OPTMIZATION_START, _logger)
            self.subscribe(Events.OPTMIZATION_STEP, _logger)
            self.subscribe(Events.OPTMIZATION_END, _logger)
            self.subscribe(Events.BATCH_END, _logger)
    
    def partner_register(self,params,clear=False):
        '''register point with target of -1'''
        if clear: self.partner_space.clear()
        self.partner_space.register(params)
        
    def fit_gp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
    
    def constrained_rng(self,n_points,bin=False):
        '''
        Random number generator that deals more effectively with highly constrained spaces. 
        
        Works only off single constraint of form L - sum(x_i) >=0
        Where the lower bound of each x_i is 0. 
        Parameters
        ----------
        n_points: integer number of points to generate
        '''
        if len(self.constraints) != 1:
            raise ValueError("Too many constraints for constrained random number generator")
        
        bounds = self.space.bounds
        steps = self.space.steps
        random_state = self._random_state
        
        # Get size and max amount from single constraint
        s = self.constraints[0]
        p = re.compile('(\d+)\]<0.5')
        ms = p.findall(s)
        # Count variables and constrained variables
        n_var = self.space.dim
        n_constrained_var= 0 
        for i in range(n_var):
            if 'x[{}]'.format(i) in s: n_constrained_var +=1
        # Initialize randoms
        assert bounds.shape[0] == n_var, "Phil's an asshole. Check your understanding of your code" #TEST
        x = np.zeros((n_points, n_var))
        # Get max value of liquid constraint
        try:
            max_val = float(s.split(' ')[0])
        except:
            raise SyntaxError("Is your liquid constraint lead by the max volume? : {}".format(s))
        
        # Generator for complements that are consistent with max
        # followed by simplex sampling for constrained
        # followed by random sampling for unconstrained
        if ms:
            complements = [int(m) for m in ms]
            for i in range(n_points):
                rem_max_val = -1 
                while rem_max_val<=0:
                    rem_max_val = max_val
                    for complement in complements:
                        x[i,complement] = random_state.uniform(bounds[complement, 0], bounds[complement, 1])
                        # Extract regex
                        p = re.compile('- \(\(x\[{:d}\]<0.5\) \* \(\(\(0.5 - x\[{:d}\]\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \) - '
                                       '\(\(x\[{:d}+\]>=0.5\) \* \(\(\(x\[{:d}\] - 0.5\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \)'.format(complement,complement,complement,complement))
                        reduction = p.findall(s)[0]
                        rem_max_val += pd.eval(reduction,local_dict={'x':x[i,:]})
                rnd = get_rnd_quantities(rem_max_val,n_constrained_var,random_state)
                cnt = 0
                for j in range(n_var): 
                    if j in complements:
                        continue
                    elif 'x[{}]'.format(j) in self.constraints[0]:
                        x[i,j]  = min(rnd[cnt],bounds[j, 1])
                        cnt+=1
                    else:
                        x[i,j] = random_state.uniform(bounds[j, 0], bounds[j, 1])            
        else:
            for i in range(n_points):
                cnt = 0
                rnd = get_rnd_quantities(max_val,n_constrained_var,random_state)
                for j in range(n_var):
                    if 'x[{}]'.format(j) in self.constraints[0]:
                        x[i,j]  = min(rnd[cnt],bounds[j, 1])
                        cnt+=1
                    else:
                        x[i,j] = random_state.uniform(bounds[j, 0], bounds[j, 1])
        if bin: x = np.floor((x-bounds[:,0])/steps)*steps+bounds[:,0]
        return x
        
        '''
        # The old way, preserved in comment for posterity
        n_constrained_var = 0
        max_amount = self.constraints[0].replace('-','')
        for i in range(len(self.space.keys)):
            if 'x[{}]'.format(i) in self.constraints[0]: 
                n_constrained_var+=1
                max_amount = max_amount.replace('x[{}]'.format(i),'')
        
        max_amount = float(max_amount)

        
        # Generate randoms and update sample
        x = np.zeros((n_points, bounds.shape[0]))
        for i in range(n_points):
            cnt = 0
            rnd = get_rnd_quantities(max_amount,n_constrained_var,random_state)
            for j in range(bounds.shape[0]):
                if 'x[{}]'.format(j) in self.constraints[0]:
                    x[i,j]  = min(rnd[cnt],bounds[j, 1])
                    cnt+=1
                else:
                    x[i,j] = random_state.uniform(bounds[j, 0], bounds[j, 1])
            
        if bin: x = np.floor((x-bounds[:,0])/steps)*steps+bounds[:,0]
        return x
        '''
    
    def suggest(self, utility_function,sampler='greedy',fit_gp=True,**kwargs):
        """
        Potential keywords 
        ------------------
        n_acqs: Integer number of acquisitions to take from acquisition function ac.
        n_warmup: number of times to randomly sample the aquisition function
        n_iter: number of times to run scipy.minimize
        multiprocessing: number of cores for multiprocessing of scipy.minimize
        n_slice: number of samples in slice sampling
    
        Returns
        -------
        list length n_acqs of dictionary style parameters 
        """
        if len(self._space) == 0:
            return [self._space.array_to_params(self.space._bin(self._space.random_sample(constraints=self.get_constraint_dict()))) for _ in range(kwargs.get('n_acqs',1))]

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax(s) of the acquisition function.
        if sampler=='KMBBO':
            if self.constraints:
                suggestion = disc_constrained_acq_KMBBO(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
            else:
                suggestion = disc_acq_KMBBO(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
        elif sampler=='greedy': 
            if self.constraints:
                suggestion = disc_constrained_acq_max(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
            else:
                suggestion = disc_acq_max(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
        else:
            if self.constraints:
                suggestion = disc_constrained_acq_max(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
            else:
                suggestion = disc_acq_max(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)

        return self._space.array_to_params(suggestion)