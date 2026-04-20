import numpy as np

class ChuyenViRTheta:
    def __init__(self, U_full, nodes):
        self.U = U_full
        self.nodes = nodes

    def get_displacements(self):
        """
        Chuyển đổi Ux, Uy sang Ur (hướng kính) và U_theta (tiếp tuyến)
        """
        ux = self.U[0::2]
        uy = self.U[1::2]
        
        ur = np.zeros_like(ux)
        utheta = np.zeros_like(uy)
        
        for i, (x, y) in enumerate(self.nodes):
            # Tính góc theta của nút hiện tại
            theta = np.arctan2(y, x)
            c = np.cos(theta)
            s = np.sin(theta)
            
            # Công thức chuyển đổi ma trận xoay
            ur[i] = ux[i] * c + uy[i] * s
            utheta[i] = -ux[i] * s + uy[i] * c
            
        return ur, utheta