import numpy as np
import matplotlib.pyplot as plt
# =========================
# MESH (1/4 domain)
# =========================
class Mesh:
    def __init__(self, Ri, Ro, nr, nt, element_type, mode="quarter"):
        self.nodes = []
        self.elements = []
        self.element_type = element_type
        self.mode = mode

        r_vals = np.linspace(Ri, Ro, nr)

        if mode == "quarter":
            theta_vals = np.linspace(0, np.pi/2, nt)
        elif mode == "full":
            theta_vals = np.linspace(0, 2*np.pi, nt, endpoint=False)

        for r in r_vals:
            for t in theta_vals:
                self.nodes.append([r*np.cos(t), r*np.sin(t)])

        self.nodes = np.array(self.nodes)

        if mode == "quarter":
            nt_eff = nt
            for i in range(nr-1):
                for j in range(nt-1):
                    n1 = i*nt_eff + j
                    n2 = i*nt_eff + j+1
                    n3 = (i+1)*nt_eff + j+1
                    n4 = (i+1)*nt_eff + j

                    if element_type == "Q4":
                        self.elements.append([n1,n2,n3,n4])
                    else:
                        self.elements.append([n1,n2,n3])
                        self.elements.append([n1,n3,n4])

        else:  # full 360
            nt_eff = nt
            for i in range(nr-1):
                for j in range(nt_eff):
                    n1 = i*nt_eff + j
                    n2 = i*nt_eff + (j+1)%nt_eff
                    n3 = (i+1)*nt_eff + (j+1)%nt_eff
                    n4 = (i+1)*nt_eff + j

                    if element_type == "Q4":
                        self.elements.append([n1,n2,n3,n4])
                    else:
                        self.elements.append([n1,n2,n3])
                        self.elements.append([n1,n3,n4])

        self.ndof = len(self.nodes)*2
