'''
Subclass of Experiment that will operate on a library of results from a previous experiment. 
'''

from experiment import Experiment
from bayes_opt import DiscreteBayesianOptimization, UtilityFunction
from bayes_opt.event import Events
from kuka_parser import Parser
from time import time, sleep

class PseudoExperiment(Experiment):
    
    def __init__(self, directory_path=None, sampler='KMBBO', verbose = 0, random_state = None,
                 MINI_BATCH=None, BATCH=None, BATCH_FILES=None, SLEEP_DELAY=None):
        super(PseudoExperiment,self).__init__(directory_path)
        if MINI_BATCH: self.MINI_BATCH=MINI_BATCH
        if BATCH_FILES: self.BATCH_FILES=BATCH_FILES
        if BATCH: self.BATCH=BATCH
        if SLEEP_DELAY: self.SLEEP_DELAY=SLEEP_DELAY
        self.sampler=sampler
        self.unavailable_points = [] 
        self.set_reference_space()
        self.generate_model(verbose=verbose, random_state=random_state)
        
    def set_reference_space(self):
        '''
        Generates a partner space using the DiscreteBayesianOptimization class for querying
        
        Draws from update_points_and_targets, generates partner space, then clears lists
        '''
        # Convert self.rng to tuple format
        prange = {p:(r['lo'],r['hi'],r['res']) for p,r in self.rng.items()}
        # Initialize optimizer and space 
        dbo = DiscreteBayesianOptimization(f=None,
                                          prange=prange,
                                          verbose=0,
                                          random_state=1,
                                          constraints=self.constraints)
        print("Formulating reference space")
        self.update_points_and_targets()
        for idx, point in enumerate(self.points):
            dbo.register(params=point, target=self.targets[idx])
        
        # reinitialize parser and experiment
        self.parser = Parser(self.compounds, self.directory_path)
        self.targets = []
        self.points = []
        
        self.reference_space = dbo.space
        print("Reference space complete")
        return    
    
    def run_batch(self,batch):
        '''
        Runs a batch by checking the partner space
        '''
        for point in batch:
            if self.reference_space.params_to_array(point) in self.reference_space:
                self.points.append(point)
                target = self.reference_space.probe_discrete(point)
                self.targets.append(target)
                self.optimizer.register(params=point, target=target)
                if self.optimizer._verbose: self.optimizer.dispatch(Events.BATCH_END)
            else:
                self.unavailable_points.append(point)
        return
    
    def generate_model(self, verbose=0, random_state=None):
        '''
        Creates and returns Bayesian optimizer 
        Saves previous model in folder according to read batch number, or 0 if none is available
        Ignores potential partner space (i.e. running/runqueue folder).
        Arguments
        ----------
        verbose: 0 (quiet), 1 (printing only maxima as found), 2 (print every registered point)
        random_state: integer for random number generator
        
        Returns
        ----------
        dbo: instance of DiscreteBayesianOptimization
        '''
        prange = {p:(r['lo'],r['hi'],r['res']) for p,r in self.rng.items()}
        # Initialize optimizer 
        dbo = DiscreteBayesianOptimization(f=None,
                                          prange=prange,
                                          verbose=verbose,
                                          random_state=random_state,
                                          constraints=self.constraints)
        if verbose: 
            dbo._prime_subscriptions()
            dbo.dispatch(Events.OPTMIZATION_START)
        # Register past data to optimizer (if any)
        for idx, point in enumerate(self.points):
            dbo.register(params=point, target=self.targets[idx])
            if verbose and idx%batch_size==0: dbo.dispatch(Events.BATCH_END)
        
        self.optimizer = dbo
        return
    
    def generate_batch(self,batch_size=None, utility_kind="ucb",kappa=2.5,xi=0.0,**kwargs):
        '''
        Creates optimizer, registers all previous data, and generates a proposed batch.
        Arguments
        ----------
        batch_size: integer number of points to suggest per batch
        verbose: 0 (quiet), 1 (printing only maxima as found), 2 (print every registered point)
        random_state: integer for random number generator
        utility_kind: Utility function to use ('ucb', 'ei', 'poi')
        kappa: float, necessary for 'ucb' utility function
        xi: float, translation of gaussian function
        **kwargs: dictionary passed to suggestion function. See bayes_opt.parallel_opt.disc_acq_max() for options
        
        Returns
        ----------
        batch: list of dictionaries containing parameters for each variable in the experiment 
        '''
        if batch_size is None: batch_size = self.BATCH
        sampler=self.sampler
        batch = []
        # Convert self.rng to tuple format
        prange = {p:(r['lo'],r['hi'],r['res']) for p,r in self.rng.items()}
        # Initialize optimizer and utility function 
        utility = UtilityFunction(kind=utility_kind, kappa=kappa, xi=xi)
        # Generate batch of suggestions
        batch = self.optimizer.suggest(utility,sampler=sampler,n_acqs=batch_size,fit_gp=True,**kwargs)
        return batch

if __name__ == '__main__':
    time_limit = 60*5
    exp = PseudoExperiment('./5dye_start_ml/',sampler='KMBBO', verbose = 2)
    start = time()
    while True:
        batch = exp.generate_batch()
        exp.run_batch(batch)
        sleep(exp.SLEEP_DELAY)
        print("Size of parameter space: {}".format(len(exp.optimizer.space)))
        print("Size of partner space: {}".format(len(exp.optimizer.partner_space)))
        print("Size of reference space: {}".format(len(exp.reference_space)))
        print("Number of inaccessible points: {}".format(len(exp.unavailable_points)))
        if time()-start>time_limit: break
    if self.optimizer.verbose: self.optimizer.dispatch(Events.OPTMIZATION_END)
    print("The following points were proposed, but unavailable")
    for point in self.unavailable_points:
        print(point)
