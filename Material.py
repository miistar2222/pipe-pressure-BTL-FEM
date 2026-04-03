import numpy as np
import matplotlib.pyplot as plt
# =========================
# MATERIAL
# =========================
class Material:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.D = self.compute_D()

    def compute_D(self):
        E, nu = self.E, self.nu
        coef = E / ((1 + nu) * (1 - 2 * nu))
        return coef * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])