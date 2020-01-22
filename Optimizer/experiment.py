'''
Experiment class containing all information about the current setup of experiment.
Lists of liquids (Experiment.liq) and solids (Experiment.sol).
Allowed ranges of quantities for every component in a dict (Experiment.rng)
'''
from bayes_opt import DiscreteBayesianOptimization, UtilityFunction
from kuka_parser import Parser
import os
from time import time, sleep
import datetime
import multiprocessing
from shutil import copyfile
import pickle
import numpy as np
import traceback
import uuid
import math


class Experiment:
    MINI_BATCH = 16
    BATCH = 16
    BATCH_FILES = 1  # number of files that we want to see in the queue, should be BATCH/BATCH_FILES = MINI_BATCH
    SLEEP_DELAY = 5  # delay in seconds before querying the queue folder again

    directory_path = './'

    def __init__(self, directory_path=None):
        if directory_path: self.directory_path = directory_path

        # General setup of the experiment
        self.compounds = []  # Simply the list of compounds to vary
        self.properties = {}  # Properties (liquid, solid, etc.) of the compounds, e.g. comp['P10'] = {'phys': 'solid', 'proc' : 'cat'}
        self.rng = {}  # Ranges with resolution, e.g. rng['P10'] = {'lo' : 0, 'hi' : 1, 'res' : 0.1}
        self.dbo_ranges = {}  # Ranges with resolution formated for dbo (including maping of complements)
        self.constraints = []  # list of the constraints that points should satisfy, e.g.
        self.controls = []  # list of the control experiments to include in each minibatch
        self.complements = {}  # Mapping of all complementary variables to single dimensions in optimizer space {'!Complement!_01' : {}}

        # Outcomes of ongoing experimentation
        self.points = []  # list of measured targets (different experiments), e.g. [{'P10' : 0.1, 'TiO2' : 0.2}, ... ]
        self.targets = []  # measured response at the experiments [1.1, 2.1, ...]

        self.name = 'Unknown'  # Name of the experiment that will appear in all the files
        self.batch_number = 1  # Number of the mini_batch to submit next
        self.liquids = []  # list of liquids
        self.constants = {}  # lst of compounds to be kept constant during measurements for the current Search space
        self.identical_compounds = {}  # dictionary with compound name as key, with dictionarys  <compound, <other liquid, concentration factor>>

        self.__read_config()
        self.__prep_dirs()
        self.parser = Parser(self.compounds, self.directory_path)  # Associated parser responsible for IO operations

    def __read_config(self):
        '''
        The function reads the optimizer.config file and 
        fills in all the general parameters of the experiment
        Also, read optimizer.state to get the next batch number
        '''
        try:
            with open(self.directory_path + 'optimizer.config', "r") as f:
                # compounds list is expected first
                compounds_section = True
                constraints_section = False
                controls_section = False
                complements_section = False

                self.name = f.readline().rstrip()

                for line in f.readlines():

                    if line.startswith("#") or line.isspace():
                        continue

                    else:
                        if line.startswith("Constraints"):
                            constraints_section = True
                            compounds_section = False
                            controls_section = False
                            complements_section = False
                            continue
                        elif line.startswith("Controls"):
                            constraints_section = False
                            compounds_section = False
                            controls_section = True
                            complements_section = False
                            continue
                        elif line.startswith("Complements"):
                            constraints_section = False
                            compounds_section = False
                            controls_section = False
                            complements_section = True
                            cidx = 1
                            continue
                        if compounds_section:
                            tmp = line.rstrip().split(sep=',')
                            name = tmp[0]
                            self.compounds.append(name)
                            self.properties[name] = {'phys': tmp[1], 'proc': tmp[2]}
                            self.rng[name] = {'lo': float(tmp[3]), 'hi': float(tmp[4]), 'res': float(tmp[5])}

                            # list liquids
                            if self.properties[name]['phys'] == 'liquid':
                                self.liquids.append(name)

                            # list constants
                            if self.rng[name]['lo'] == self.rng[name]['hi']:
                                self.constants[name] = self.rng[name]['lo']

                            alt_liq = math.floor(len(tmp) / 2) - 3
                            if alt_liq > 0:
                                self.identical_compounds[name] = {}
                                for x in range(alt_liq):
                                    self.identical_compounds[name][tmp[6 + 2 * x]] = tmp[7 + 2 * x]

                        if constraints_section:
                            self.constraints.append(line.rstrip())

                        if controls_section:
                            cols = line.rstrip().split(sep=',')
                            d = {}
                            for col in cols:
                                tmp = col.split(sep=':')
                                d[tmp[0].strip()] = float(tmp[1])
                            self.controls.append(d)

                        if complements_section:
                            ''' General format:
                            {!Complement!_cidx: {A_name: compound, A_range: rng, B_name: compound, B_range: rng}}
                            '''
                            cols = [col.strip() for col in line.rstrip().split(sep=':')]
                            assert len(cols) == 2, "Complements come in pairs! Failure at '{}'".format(line)
                            assert cols[0] in self.compounds, "This complement is not in the compounds: {}".format(
                                cols[0])
                            assert cols[1] in self.compounds, "This complement is not in the compounds: {}".format(
                                cols[1])
                            self.complements['!Complement!_{}'.format(cidx)] = {'A_name': cols[0],
                                                                                'B_name': cols[1]}
                            try:
                                self.complements['!Complement!_{}'.format(cidx)]['A_range'] = self.rng[cols[0]]
                                self.complements['!Complement!_{}'.format(cidx)]['B_range'] = self.rng[cols[1]]
                            except:
                                raise SyntaxError(
                                    "Please place complements after compounds and ranges in configuration file.")
                            cidx += 1

            # Update of optimizer ranges and constraints from complements
            self.dbo_ranges = {p: (r['lo'], r['hi'], r['res']) for p, r in self.rng.items() if r['lo'] < r['hi']}
            for key, dict in self.complements.items():
                a = self.dbo_ranges.pop(dict['A_name'])
                b = self.dbo_ranges.pop(dict['B_name'])
                self.dbo_ranges[key] = (0., 1., min(a[2] / (a[1] - a[0]) / 2,
                                                    b[2] / (b[1] - b[0]) / 2))

                new_constraints = []
                for s in self.constraints:
                    s = s.replace(dict['A_name'],
                                  "(({}<0.5) * (((0.5 - {})/0.5) * ({:f}-{:f}) + {:f}) )".format(key, key, a[1], a[0],
                                                                                                 a[0]))
                    s = s.replace(dict['B_name'],
                                  "(({}>=0.5) * ((({} - 0.5)/0.5) * ({:f}-{:f}) + {:f}) )".format(key, key, b[1], b[0],
                                                                                                  b[0]))
                    new_constraints.append(s)
                self.constraints = new_constraints

        except IOError:
            print("There is no configuration file in the experiment folder.")
        try:
            with open(self.directory_path + 'optimizer.state', "r") as f:
                self.batch_number = int(f.readline().rstrip())

        except IOError:
            print("No state file present. Generating new one.")
            with open(self.directory_path + 'optimizer.state', "w") as f:
                f.write('1')

    def __prep_dirs(self):
        '''
        Prepare all necessary directories
        '''
        dirs = ['running', 'runqueue', 'completed', 'models']
        for d in dirs:
            try:
                os.makedirs(os.path.join(self.directory_path, d), exist_ok=True)
            except:
                raise

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __str__(self):
        output = f"Experiment '{self.name}' is in progress.\n"
        output += f"The next batch is {self.batch_number}\n"
        output += "Compounds to vary with ranges and resolution:\n"
        for composition, bounds in self.rng.items():
            # print(bounds)
            output += (f'    {composition}: [' + str(bounds['lo']) \
                       + ', ' + str(bounds['hi']) + ', ' + str(bounds['res']) + ']\n')
        return output

    def complement_mapping(self, point):
        '''
        Maps any complementary variables in a point to a single variable in optimizer space, 
        or maps the  single variable in optimizer space back to the complementary points
        in configuration space. 
        
        Arguments
        ----------
        point: dictionary of variable names and points, updated inplace
        '''
        if len(self.complements) == 0: return

        keys = [key for key in point]
        if any(key.split('_')[0] == '!Complement!' for key in keys):
            for key in keys:
                if key.split('_')[0] != '!Complement!': continue
                dict = self.complements[key]
                val = point.pop(key)
                if val < 0.5:
                    a_val = ((0.5 - val) / 0.5) * (dict['A_range']['hi'] - dict['A_range']['lo']) + dict['A_range'][
                        'lo']
                    b_val = 0
                else:
                    a_val = 0
                    b_val = ((val - 0.5) / 0.5) * (dict['B_range']['hi'] - dict['B_range']['lo']) + dict['B_range'][
                        'lo']

                point[dict['A_name']] = a_val
                point[dict['B_name']] = b_val
        else:
            for complement, dict in self.complements.items():
                a_val = point.pop(dict['A_name'])
                b_val = point.pop(dict['B_name'])
                if a_val > 0 and b_val > 0: raise RuntimeError("Complementary values are both nonzero")
                if a_val > 0:
                    new_val = 0.5 - 0.5 * (
                                (a_val - dict['A_range']['lo']) / (dict['A_range']['hi'] - dict['A_range']['lo']))
                elif b_val > 0:
                    new_val = 0.5 + 0.5 * (
                                (b_val - dict['B_range']['lo']) / (dict['B_range']['hi'] - dict['B_range']['lo']))
                else: #Zero case
                    new_val = 0.5
                point[complement] = new_val

        return


    def output_space(self, path):
        """
        Outputs complete space as csv file.
        Simple function for testing
        Parameters
        ----------
        path

        Returns
        -------

        """
        import pandas as pd
        df = pd.DataFrame(self.points)
        df['Target'] = self.targets
        df.to_csv(path)

    def clear_previous_model(self):
        '''
        Moves previous model to past model folder
        data = {}  # Data dictionary to be saved

        '''
        fname = os.path.join(self.directory_path, 'optimizer.pickle')
        if os.path.isfile(fname):
            copyfile(fname,
                     os.path.join(self.directory_path, 'models', 'state_{}.pickle'.format(self.batch_number - 1)))
            os.remove(fname)

    def generate_model(self, verbose=0, random_state=None):
        '''
        Creates, saves, and returns Bayesian optimizer 
        Saves previous model in folder according to read batch number, or 0 if none is available
        Arguments
        ----------
        verbose: 0 (quiet), 1 (printing only maxima as found), 2 (print every registered point)
        random_state: integer for random number generator
        
        Returns
        ----------
        dbo: instance of DiscreteBayesianOptimization
        '''

        self.clean_queue()
        data = {}  # Data dictionary to be saved

        prange = self.dbo_ranges
        # Initialize optimizer and utility function 
        dbo = DiscreteBayesianOptimization(f=None,
                                           prange=prange,
                                           verbose=verbose,
                                           random_state=random_state,
                                           constraints=self.constraints)
        if verbose:
            dbo._prime_subscriptions()
            dbo.dispatch(Events.OPTMIZATION_START)

        # Register past data to optimizer
        self.update_points_and_targets()
        for idx, point in enumerate(self.points):
            dbo.register(params=point, target=self.targets[idx])
            if verbose and idx % batch_size == 0: dbo.dispatch(Events.BATCH_END)

        # Register running data to partner space in optimizer
        running_points = self.get_running_points()
        for idx, point in enumerate(running_points):
            if idx == 0:
                dbo.partner_register(params=point, clear=True)
            else:
                dbo.partner_register(params=point, clear=False)

        # Fit gaussian process
        data['random_state'] = np.random.get_state()
        if len(dbo.space) > 0:
            dbo.output_space('dbo_space.csv')
            #self.output_space('exp_space.csv')
            start_time = time()
            dbo.fit_gp()
            print("Model trained in {:8.2f} minutes".format((time()-start_time)/60))
            if any(dbo._gp.kernel_.k1.k1.length_scale<5e-3):
                print("Warning: Very short length scale detected when fitting Matern kernel. Retraining model...")
                start_time = time()
                dbo.fit_gp()
                print("Model trained in {:8.2f} minutes".format((time() - start_time) / 60))
            if any(dbo._gp.kernel_.k1.k1.length_scale>5e2):
                print("Warning: Very long length scale detected when fitting Matern kernel.")
            print("Model length scales:")
            for key, value in dict(zip(dbo.space.keys, dbo._gp.kernel_.k1.k1.length_scale)).items():
                print("{}: {:8.4f}".format(key, value))
            print("Model noise: {}".format(dbo._gp.kernel_.k2.noise_level))
            print("Model constant scale: {}".format(dbo._gp.kernel_.k1.k2.constant_value))
        # Refresh queue and copy old model
        self.read_batch_number()
        fname = os.path.join(self.directory_path, 'optimizer.pickle')
        if os.path.isfile(fname):
            copyfile(fname,
                     os.path.join(self.directory_path, 'models', 'state_{}.pickle'.format(self.batch_number - 1)))

        # Build dictionary to save and return model
        data['processed_files'] = list(self.parser.processed_files.keys())
        data['model'] = dbo
        data['uuid'] = uuid.uuid4()
        with open(fname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return dbo

    def generate_batch(self, batch_size=BATCH, verbose=0, random_state=None, utility_kind="ucb", kappa=2.5, xi=0.0,
                       sampler='greedy', **kwargs):
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
        batch = []
        # Update kwargs 
        if sampler == 'greedy' or sampler == 'capitalist':
            kwargs['complements'] = bool(self.complements)
        # Initialize optimizer and utility function 
        fname = os.path.join(self.directory_path, 'optimizer.pickle')
        if os.path.isfile(fname):
            with open(fname, 'rb') as handle:
                data = pickle.load(handle)
                dbo = data['model']
                running_points = self.get_running_points()
                for point in running_points:
                    dbo.partner_register(params=point, clear=False)
                self.model_uuid = data['uuid']
        else:
            dbo = self.generate_model(verbose=verbose, random_state=random_state)
            self.model_uuid = self.get_saved_model_uuid()
        utility = UtilityFunction(kind=utility_kind, kappa=kappa, xi=xi)

        # Generate batch of suggestions
        dbo.reset_rng()
        batch = dbo.suggest(utility, sampler=sampler, n_acqs=batch_size, fit_gp=False, **kwargs)

        # Clear and re-register running data to partner space in optimizer (can be adjusted in capitalist)
        running_points = self.get_running_points()
        for idx, point in enumerate(running_points):
            if idx == 0:
                dbo.partner_register(params=point, clear=True)
            else:
                dbo.partner_register(params=point, clear=False)
        for point in batch:
            self.complement_mapping(point)
        return batch

    def register_mini_batch(self, mini_batch):
        '''
        Submit the mini_batch to the workflow.
        '''
        self.read_batch_number()
        if len(mini_batch) != self.MINI_BATCH:
            print("Warning! You are not submitting the right amount of measurements per mini-batch.")

        batch_name = self.name + '-' + "{:0>4d}".format(self.batch_number)
        self.parser.submit_mini_batch(batch_name, mini_batch, self.liquids)
        self.batch_number += 1

        self.write_batch_number()

    def queue_size(self):
        '''
        Simply checks the number of files available in the queue
        Note:  (1) I use 'exp_name in first_line' to check whether
            the file belongs to our experiment. Thus, it is better
            to use some fixed prefix for ml-driven runs
        '''
        queue_size = 0
        folder_path = self.directory_path + 'runqueue/'
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1] == '.run':
                try:
                    with open(os.path.join(folder_path, f), "r") as file_input:
                        if self.name in file_input.readline():
                            queue_size += 1

                except IOError:
                    print("One of the queue files was not processed. Check Experiment.queue_size.")
                except UnicodeDecodeError:
                    print("Unreadable files in queue. Potentially system files polluting space.")

        return queue_size

    def write_batch_number(self):
        '''writes out file for state tracking'''
        try:
            with open(self.directory_path + 'optimizer.state', "w") as f:
                f.write(str(self.batch_number))
        except IOError:
            print("Failed to save the batch number in the optimizer.state.")
            print(f"Current batch number is {self.batch_number}")

    def read_batch_number(self):
        '''reads state file for updated batch number'''
        try:
            with open(self.directory_path + 'optimizer.state', "r") as f:
                self.batch_number = int(f.readline().rstrip())

        except IOError:
            print("No state file present. Generating new one.")
            self.batch_number = 1
            self.write_batch_number()

    def clean_queue(self):
        '''
        This will clear the queue of experimental files with the experiment name. 
        Used primarily when a fresh model is generated, during active workflow time. 
        '''
        self.read_batch_number()
        folder_path = os.path.join(self.directory_path, 'runqueue/')
        for f in os.listdir(folder_path):
            clean = False
            if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1] == '.run':
                try:
                    with open(os.path.join(folder_path, f), "r") as file_input:
                        if self.name in file_input.readline() \
                                and self.name + '-0' not in file_input.readline():  # not deleting manually submitted files
                            clean = True
                except IOError:
                    print("One of the queue files was not processed ({:s}). Check Experiment.clean_queue.".format(f))
                except UnicodeDecodeError:
                    print("Unreadable files in queue. Potentially system files polluting space.")
                if clean:
                    os.remove(os.path.join(folder_path, f))
                    self.batch_number -= 1
        print("The queue has been cleared.\n", end='Time is ')
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.write_batch_number()

    def get_running_points(self):
        '''
        Check whether there are experiments in the runque or active running, and return list. 
        This purposefully ignores '_dispensed' values, since this shouldn't be relevant until completed. 
        '''
        dfs = self.parser.process_running(self.name)
        skipped = 0
        _points = []
        for df in dfs:
            for idx, row in df.iterrows():
                point = {}

                if self.skip_point(row):
                    skipped = skipped + 1
                else:
                    for comp in self.compounds:
                        if self.rng[comp]['lo'] < self.rng[comp]['hi']:
                            if comp in df:
                                point[comp] = row[comp]
                                continue
                            else:
                                if comp in self.identical_compounds.keys():
                                    found_alternative = False
                                    for alternative in self.identical_compounds[comp].keys():
                                        if alternative in df:
                                            # print(
                                            #    'found_alternative ' + alternative + ' in ' + row['Name'] +
                                            # ' for compound' + comp + ' with final concentration ' + str(
                                            #        float(row[alternative]) * float(
                                            #            self.identical_compounds[comp][alternative])))
                                            found_alternative = True
                                            point[comp] = float(row[alternative]) * float(
                                                self.identical_compounds[comp][alternative])
                                            continue
                                    if not found_alternative:
                                        point[comp] = 0
                                else:
                                    point[comp] = 0
                                point[comp] = 0
                    self.complement_mapping(point)
                    _points.append(point)

        if skipped != 0:
            print('Warning: Ignored ' + str(skipped) + ' points in running folder.')
        return _points

    def skip_point(self, point):
        '''
                    Exclude any points that contain compounds that are not under consideration:
                    i.e. filter non-variable compounds that are != 0
                    Returns false if there is any compounds in the sample that are not under consideration
        '''
        for key, value in point.iteritems():
            # case 0: sample is leaking
            if key == 'oxygen_evolution_micromol' and value > 5:
                print('Warning, skipping leaky point ' + point['Name'])
                return True

            # case 1: standard column that is not representing a compound
            if (key in {'SampleIndex', 'SampleNumber', 'Name', 'vial_capped', 'gc_well_number',
                        'hydrogen_evolution', 'oxygen_evolution', 'hydrogen_evolution_micromol',
                        'oxygen_evolution_micromol', 'water', 'water_dispensed',
                        'internal_hydrogen_standard_micromol', 'weighted_hydrogen_micromol', 'sample_location_weight',
                        'weighted_is_sl_hydrogen_evolution_micromol'}) \
                    or 'Unnamed' in key:  # deal with faulty comma
                continue

            # case 2: column directly representing variable
            if (key in self.compounds) or ((len(key) > 10) and (key[:-10] in self.compounds)):
                # case 2.5: column representing control only/ compound not included in experiment.
                if key in self.rng and self.rng[key]['hi'] <= 0 and value>0:
                    #print('Warning, ignoring point with ' + key + ' and value ' + str(value))
                    return True
                else:
                    continue

            # case 3: column representing variable but with different concentration for compound in list
            skip = False;
            for compound in self.compounds:
                if compound in self.identical_compounds and \
                        ((key in self.identical_compounds[compound].keys()) or (
                                (len(key) > 10) and (key[:-10] in self.identical_compounds[compound].keys()))):
                    # print('Found compound that is a variable, but with different concentration' + key)
                    skip = True
                    continue
            if skip:
                continue

            # case 4: column is unclear, but value is 0
            if value == 0:
                continue

            # case 5: column not representing variable compound in list
            # print('Warning, ignoring point with ' + key + ' and value ' + str(value))
            return True

        # no unexpected columns found
        return False

    def update_points_and_targets(self):
        '''
        Check whether there are new available measurements.
        If there are, then fill in self.points and self.targets

        Note (1) 'Name_dispensed' has preference over simply 'Name'
             (2) Silently ignores values of all other compounds!
             (3) For now we kick out rows with Nan's in 'hydrogen_evolution'

        '''
        for filename in self.parser.process_completed_folder(self.name):
            # print(filename)
            frame = self.parser.processed_files[filename]
            frame.dropna(subset=['hydrogen_evolution'],
                         inplace=True)  # Update on a later date for a more appropriate handling
            # print(filename, self.parser.processed_files[filename].tail())
            print(f"Adding data from {filename} to the list of points: {len(frame)} measurements.")

            f_targets = list(self.optimisation_target(frame))
            skipped = 0
            for idx, row in frame.iterrows():

                point = {}
                skip_point = self.skip_point(row)
                if skip_point:
                    skipped = skipped + 1
                else:
                    for comp in self.compounds:
                        if self.rng[comp]['lo'] < self.rng[comp]['hi']:
                            if comp + '_dispensed' in frame:
                                point[comp] = row[comp + '_dispensed']
                                continue

                            if comp in frame:
                                point[comp] = row[comp]
                                continue

                            if comp in self.identical_compounds.keys():
                                found_alternative = False
                                for alternative in self.identical_compounds[comp].keys():
                                    # there seem to be nan values if the batch has any comments => ignore them
                                    if (alternative + '_dispensed') in frame and not math.isnan(row[alternative]):
                                        # print(
                                        #    'found_alternative ' + alternative + ' in ' + row['Name'] +
                                        #    ' for compound' + comp + ' with final value ' +
                                        #    str(float(row[alternative+'_dispensed'])
                                        #        * float(self.identical_compounds[comp][alternative])))
                                        found_alternative = True
                                        point[comp] = float(row[alternative + '_dispensed']) * float(
                                            self.identical_compounds[comp][alternative])
                                        continue
                                    elif alternative in frame and not math.isnan(row[alternative]):
                                        # print(
                                        #    'found_alternative ' + alternative + ' in ' + row['Name'] +
                                        #    ' for compound' + comp +  ' with final value ' + str(
                                        #        float(row[alternative]) * float(self.identical_compounds[comp][alternative])))
                                        found_alternative = True
                                        point[comp] = float(row[alternative]) * float(
                                            self.identical_compounds[comp][alternative])
                                        continue

                                if not found_alternative:
                                    point[comp] = 0
                            else:
                                point[comp] = 0
                            # print(f"Warning! {comp} was not found in the file {filename}")
                    self.complement_mapping(point)
                    # ### TEST BLOCK ###
                    # print("Using test block in update_points_and_targets. This should not be in deployment")
                    # if np.random.uniform() < 1:
                    #     self.points.append(point)
                    #     self.targets.append(f_targets[idx])
                    # ### TEST BLOCK ###
                    ### REAL BLOCK ###
                    self.points.append(point)
                    self.targets.append(f_targets[idx])
                    ### REAL BLOCK ###

            if skipped != 0:
                print('Warning: Ignored ' + str(skipped) + ' points.')
            assert len(self.targets) == len(self.points), "Missmatch in points and targets. "\
                                                          "Error in Experiment.update_points_and_targets"
            print('Total number of points in model: ' + str(len(self.points)))

    def optimisation_target(self, frame):
        return frame['hydrogen_evolution_micromol']

    def new_model_available(self):
        new_uuid = self.get_saved_model_uuid()
        return not (self.model_uuid == new_uuid)

    def get_saved_model_uuid(self):
        fname = os.path.join(self.directory_path, 'optimizer.pickle')
        if os.path.isfile(fname):
            with open(fname, 'rb') as handle:
                data = pickle.load(handle)
                new_uuid = data['uuid']
                return new_uuid;
        return uuid.uuid4()


def clean_and_generate(exp, batches_to_generate, multiprocessing=1, perform_clean=False, sampler='greedy'):

    if (perform_clean):
        exp.clean_queue()

    KMBBO_args = {'multiprocessing': multiprocessing,
                  'n_slice': 500}
    greedy_args = {'multiprocessing': multiprocessing,
                   'n_iter': 500,
                   'n_warmup': 10000,
                   'kappa': 1.5}
    capitalist_args = {'multiprocessing': multiprocessing,
                       'exp_mean': 2.5,
                       'n_splits': 14,
                       'n_iter': 250,
                       'n_warmup': 1000
                       }

    start_time = time()
    ### Choose your own adventure ###
    if sampler == 'KMBBO':
        batch = exp.generate_batch(batch_size=batches_to_generate * (exp.MINI_BATCH - len(exp.controls)),
                                   sampler='KMBBO', **KMBBO_args)
    elif sampler == 'greedy':
        batch = exp.generate_batch(batch_size=batches_to_generate * (exp.MINI_BATCH - len(exp.controls)),
                                   sampler='greedy', **greedy_args)
    elif sampler == 'capitalist':
        batch = exp.generate_batch(batch_size=batches_to_generate * (exp.MINI_BATCH - len(exp.controls)),
                                   sampler='capitalist', **capitalist_args)
    else:
        raise ValueError("No sampler named {}".format(sampler))

    print("Batch was generated in {:.2f} minutes. Submitting.\n".format((time() - start_time) / 60))

    # add constants
    for i in range(len(batch)):
        batch[i].update(exp.constants)
    for i in range(batches_to_generate):
        exp.register_mini_batch(batch[i * (exp.MINI_BATCH - len(exp.controls)):(i + 1) * (
                exp.MINI_BATCH - len(exp.controls))] + exp.controls)


def watch_completed(lag_time=900):
    '''
    Monitors completed folder, and generates model with a lag time
    Arguments
    --------
    lag_time: interger, seconds to wait after newly discovered file to generate model
        This lag should be greater than the lag between completed MINI_BATCHES in a BATCH. 
    '''
    exp = Experiment()
    completed_dir = os.path.join(exp.directory_path, 'completed')
    n_files = 0
    for f in os.listdir(completed_dir):
        if os.path.isfile(os.path.join(completed_dir, f)): n_files += 1
    # Automatically generate model at restart
    exp.clear_previous_model()

    while True:
        count = 0
        for f in os.listdir(completed_dir):
            if os.path.isfile(os.path.join(completed_dir, f)):
                count += 1

        if count > n_files:
            print("New completed files detected. Waiting {} seconds to train new model.".format(lag_time))
            n_files = count
            sleep(lag_time)

            exp.generate_model()
            print(
                "New model trained. Old model has been saved as ./models/state_{}.pickle".format(exp.batch_number - 1))
        sleep(Experiment.SLEEP_DELAY)


def watch_queue(multiprocessing=1, sampler='greedy'):
    '''
    Monitors runqueue folder, and generates a batch based on existing model 
    or creates a fresh model if none exists. 
    '''
    exp = Experiment()
    exp.model_uuid = exp.get_saved_model_uuid()

    while True:
        # case 1: not enough batches in queue
        if exp.queue_size() < exp.BATCH_FILES:
            print("There are less than required files in the queue. Generating a new batches.\n", end='Time is ')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            missing_files = exp.BATCH_FILES - exp.queue_size()
            clean_and_generate(exp, missing_files, multiprocessing, False, sampler)
        # case 2: new model
        elif exp.new_model_available():
            print("A new model has been generated. Generating new batches.\n", end='Time is ')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            clean_and_generate(exp, exp.BATCH_FILES, multiprocessing, True, sampler)
        sleep(Experiment.SLEEP_DELAY)


if __name__ == "__main__":
    # try:
    #     p1 = multiprocessing.Process(target=watch_completed, args=(360,)) #Delay for model building when finding new data
    #     p1.start()
    #     sleep(Experiment.SLEEP_DELAY)
    #     p2 = multiprocessing.Process(target=watch_queue, args=(7,'capitalist',)) #CPUs used for batch generation and sampler choice, Search strategy
    #     p2.start()
    # except:
    #     tb = traceback.format_exc()
    #     print(tb)

    #     ### DEBUGINING LINES ###
    #     p1 = multiprocessing.Process(target=watch_completed, args=(900,)) #Delay for model building when finding new data
    #     p1.start()
    #     sleep(Experiment.SLEEP_DELAY)
    #     p2 = multiprocessing.Process(target=watch_queue, args=(4,'KMBBO',)) #CPUs used for batch generation
    #     p2.start()
    # ## IN SERIAL ###
    try:
        os.remove('optimizer.pickle')  # Clean start
    except OSError:
        pass
    watch_queue(1, 'capitalist')
    ## DEBUGING LINES ###
