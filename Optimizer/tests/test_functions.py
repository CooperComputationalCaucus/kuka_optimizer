import numpy as np


class Parabolic():
    def __init__(self, name, DIM):
        self.DIM = DIM
        self.arg_opt = np.random.random_sample((DIM,))
        self.arg_opt = np.around(self.arg_opt, decimals=3)
        self.sensitivity = np.random.random_sample((DIM,))
        self.sensitivity = np.around(self.sensitivity * 10, decimals=3)

        func_dict = {'joint': self.parabola_joint,
                     'independent': self.parabola_independent
                     }
        try:
            self.f = func_dict[name]
        except KeyError:
            raise KeyError('Function {} unavailable'.format(name))

    def parabola_joint(self, **kwargs):
        point = np.zeros(self.DIM)
        for i in range(self.DIM):
            point[i] = kwargs[f'x_{i}']

        return self.DIM - 5 * np.linalg.norm(point - self.arg_opt)

    def parabola_independent(self, **kwargs):
        point = np.zeros(self.DIM)
        for i in range(self.DIM):
            point[i] = kwargs[f'x_{i}']

        return self.DIM - sum([self.sensitivity[i] * (point[i] - self.arg_opt[i]) ** 2 for i in range(self.DIM)])


class Branin():
    '''
    Plot https://www.sfu.ca/~ssurjano/branin.html
    The function has three gloval minima equal to 0.397887 
    at (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475)
    '''

    def __init__(self):
        # Typical ranges and parameters
        self.prange = {'x1': (-5, 10, 0.125),
                       'x2': (0., 15., 0.125)}
        self.a = 1
        self.b = 5.1 / (4 * np.math.pi ** 2)
        self.c = 5 / (np.math.pi)
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.math.pi)
        self.opt = 0.397887
        self.x_opt = [(-np.math.pi, 12.275), (np.math.pi, 2.275), (9.42478, 2.475)]

    def f(self, **kwargs):
        x1 = kwargs['x1']
        x2 = kwargs['x2']
        return self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.r) ** 2 + self.s * (
                1 - self.t) * np.math.cos(x1) + self.s


class Hartmann():
    '''
    The description is at https://www.sfu.ca/~ssurjano/hart6.html
    The function has six local minima
    Global minimum is 3.32237 at
    (0.20169, 0.15011, 0.476874, 0.275332, 0.311652, 0.6573)
    '''

    def __init__(self):
        # Typical ranges and parameters
        self.prange = {'x1': (0, 1, 0.05),
                       'x2': (0, 1, 0.05),
                       'x3': (0, 1, 0.05),
                       'x4': (0, 1, 0.05),
                       'x5': (0, 1, 0.05),
                       'x6': (0, 1, 0.05)}

        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])

        self.A = np.array(
            [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]])

        self.P = 10 ** (-4) * np.array(
            [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]])

    def f(self, **kwargs):
        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = kwargs['x' + str(jj + 1)]
                Aij = self.A[ii, jj]
                Pij = self.P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2

            new = self.alpha(ii) * np.math.exp(-inner)
            outer = outer + new

        return -(2.58 + outer) / 1.94


class PhilsFun():
    '''Specific analytical funciton used for 1-d visialuation'''

    def __init__(self):
        self.f = self.fun

    def fun(self, x):
        return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1) + 2 * np.exp(
            -(x - 5) ** 2) + np.exp(-(x - 8) ** 2 / 10) + np.exp(-(x - 10) ** 2)


class GPVirtualModel():
    """
    This model uses an sklearn GP object from a pickle as a black box function.

    """

    def __init__(self, path=None, noisy=False):
        import pickle
        """
        Parameters
        ----------
        path: string, path to pickle. If none, default is used (not operating system independent). 
        """
        self.var_map = {'AcidRed871_0gL': 0,
                        'L-Cysteine-50gL': 1,
                        'MethyleneB_250mgL': 2,
                        'NaCl-3M': 3,
                        'NaOH-1M': 4,
                        'P10-MIX1': 5,
                        'PVP-1wt': 6,
                        'RhodamineB1_0gL': 7,
                        'SDS-1wt': 8,
                        'Sodiumsilicate-1wt': 9}
        if path is None:
            path = 'GPVirtualModel.pkl'
        with open(path, 'rb') as file:
            dbo = pickle.load(file)['model']
            gp = dbo._gp
        self.f = self.func
        self.gp = gp
        self.noisy = noisy

    @property
    def dim(self):
        return self.gp.X_train_.shape[1]

    @property
    def exp_max(self):
        return np.max(self.gp.y_train_)

    @property
    def param_max(self):
        return self.gp.X_train_[np.argmax(self.gp.y_train_), :]

    def func(self, **point):
        x = np.zeros(len(self.var_map))
        for key, idx in self.var_map.items():
            x[idx] = point[key]
        if self.noisy:
            return self.gp.predict(x.reshape(1,-1))[0] + np.random.uniform(-.5, .5)
        else:
            return self.gp.predict(x.reshape(1, -1))[0]


class HERVirtualModel():
    """
    This is a strictly linear model developed by AIC for an example system.
    It serves as a quick and dirty approximation to the behavior expected.
    Takes care of complement mapping for specific test case
    """

    def __init__(self):
        self.f = self.fun
        self.coeff = np.array([3.2, 0.1, 0.2, -0.1, 0.8, -1.8, 5, -5, -8, -16])
        self.var_map = {'Cysteine': 0,
                        'NaCl': 1,
                        'NaOH': 2,
                        'HCl': 3,
                        'PVP': 4,
                        'SDS': 5,
                        'Na_Silicate': 6,
                        'Acid_Red': 7,
                        'Rhodamine': 8,
                        'Methylene_Blue': 9
                        }
        self.prange = {'Cysteine': (0., 5., 0.25),
                       'NaCl': (0., 5., 0.25),
                       'NaOH': (0., 5., 0.25),
                       'HCl': (0., 5., 0.25),
                       'PVP': (0., 5., 0.25),
                       'SDS': (0., 5., 0.25),
                       'Na_Silicate': (0., 5., 0.25),
                       'Acid_Red': (0., 5., 0.25),
                       'Rhodamine': (0., 5., 0.25),
                       'Methylene_Blue': (0., 5., 0.25)
                       }
        self.constraints = [
            "5 - Cysteine - NaCl - NaOH - HCl - PVP - SDS - Na_Silicate - Acid_Red - Rhodamine - Methylene_Blue"]
        self.complements = {"pH_complement": {"A_name": "NaOH",
                                              "A_range": self.prange["NaOH"],
                                              "B_name": "HCl",
                                              "B_range": self.prange["HCl"]}}
        self.adjust_for_complement()

    def adjust_for_complement(self):
        for key, dict in self.complements.items():
            a = self.prange.pop(dict['A_name'])
            b = self.prange.pop(dict['B_name'])
            self.prange[key] = (0., 1., min(a[2] / (a[1] - a[0]) / 2,
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

    def complement_mapping(self, **point):
        """

        Parameters
        ----------
        point: dictionary of point keys

        Returns
        -------
        x: numpy array after complement mapping and key to value mapping
        """
        dict = self.complements['pH_complement']
        val = point.pop('pH_complement')
        if val < 0.5:
            pass  # A
            a_val = ((0.5 - val) / 0.5) * (dict['A_range'][1] - dict['A_range'][0]) + dict['A_range'][0]
            b_val = 0
        else:
            a_val = 0
            b_val = ((val - 0.5) / 0.5) * (dict['B_range'][1] - dict['B_range'][0]) + dict['B_range'][0]
        point[dict['A_name']] = a_val
        point[dict['B_name']] = b_val

        x = np.zeros(len(self.var_map))
        for key, idx in self.var_map.items():
            x[idx] = point[key]

        return x

    def fun(self, **kwargs):
        """

        Parameters
        ----------
        point: dictionary containing keys:
            Cysteine
            NaCl
            pH
            PVP
            SDS
            Na_Silicate
            Acid Red
            Rhodamine
            Methylene Blue

        Returns
        -------
        Linear combination of coefficients and mapped values
        """
        x = self.complement_mapping(**kwargs)
        return np.dot(self.coeff, x)
