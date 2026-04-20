import numpy as np

class NodeGenerator:
    def __init__(self, Ri, Ro, nr, nt, mode="quarter"):
        self.Ri = Ri
        self.Ro = Ro
        self.nr = nr
        self.nt = nt
        self.mode = mode
        self.nodes = self.generate_nodes()

    def generate_nodes(self):
        r_vals = np.linspace(self.Ri, self.Ro, self.nr)
        
        if self.mode == "quarter":
            theta_vals = np.linspace(0, np.pi/2, self.nt)
        else: # full 360
            theta_vals = np.linspace(0, 2*np.pi, self.nt, endpoint=False)

        nodes = []
        for r in r_vals:
            for t in theta_vals:
                nodes.append([r * np.cos(t), r * np.sin(t)])
        
        return np.array(nodes)