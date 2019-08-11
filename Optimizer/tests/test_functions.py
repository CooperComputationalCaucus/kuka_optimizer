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


class PhilsFun():
    '''Specific analytical funciton used for 1-d visialuation'''

    def __init__(self):
        self.f = self.fun

    def fun(self, x):
        return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1) + 2 * np.exp(
            -(x - 5) ** 2) + np.exp(-(x - 8) ** 2 / 10) + np.exp(-(x - 10) ** 2)


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
        for key,idx in self.var_map.items():
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
