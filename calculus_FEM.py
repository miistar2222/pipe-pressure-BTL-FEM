import numpy as np

class Analytical:
    def __init__(self, Ri, Ro, E, nu, pi, po=0.0):
        self.Ri = Ri
        self.Ro = Ro
        self.E = E
        self.nu = nu
        self.pi = pi
        self.po = po 

    def get_radial_displacement(self, r):
        r = np.where(r == 0, 1e-10, r)
        term1 = (self.pi * self.Ri**2 - self.po * self.Ro**2) / (self.Ro**2 - self.Ri**2)
        term2 = ((self.pi - self.po) * self.Ri**2 * self.Ro**2) / (self.Ro**2 - self.Ri**2)
        ur = ((1 + self.nu) / self.E) * (term1 * (1 - 2 * self.nu) * r + term2 / r)
        return ur

    def get_stresses(self, r):
        r = np.where(r == 0, 1e-10, r)
        term1 = (self.pi * self.Ri**2 - self.po * self.Ro**2) / (self.Ro**2 - self.Ri**2)
        term2 = ((self.pi - self.po) * self.Ri**2 * self.Ro**2) / (self.Ro**2 - self.Ri**2)
        
        sr = term1 - term2 / (r**2)
        st = term1 + term2 / (r**2)
        return sr, st