import numpy as np


class Parabolic():
    def __init__(self,name,DIM):
        self.DIM = DIM
        self.arg_opt = np.random.random_sample((DIM,))
        self.arg_opt = np.around(self.arg_opt, decimals=3)
        self.sensitivity = np.random.random_sample((DIM,))
        self.sensitivity = np.around(self.sensitivity*10, decimals=3)
        
        func_dict = {'joint':self.parabola_joint,
                     'independent' : self.parabola_independent
                     }
        try:
            self.f = func_dict[name]
        except:
            raise KeyError('Function {} unavailable'.format(name))
    
    def parabola_joint(self, **kwargs):
        point = np.zeros(self.DIM)
        for i in range(self.DIM):
            point[i] = kwargs[f'x_{i}']
    
        return self.DIM - 5*np.linalg.norm(point - self.arg_opt)

    def parabola_independent(self,**kwargs):
        point = np.zeros(self.DIM)
        for i in range(self.DIM):
            point[i] = kwargs[f'x_{i}']
    
        return self.DIM - sum([self.sensitivity[i]*(point[i]-self.arg_opt[i])**2 for i in range(self.DIM)])