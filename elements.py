import numpy as np
import matplotlib.pyplot as plt

# =========================
# ELEMENTS
# =========================
class ElementQ4:
    def __init__(self, mat):
        self.mat = mat

    def stiffness(self, coords):
        Ke = np.zeros((8,8))
        D = self.mat.D
        gauss = [-1/np.sqrt(3), 1/np.sqrt(3)]

        for xi in gauss:
            for eta in gauss:
                dN_dxi = np.array([
                    [-(1-eta), -(1-xi)],
                    [(1-eta), -(1+xi)],
                    [(1+eta), (1+xi)],
                    [-(1+eta), (1-xi)]
                ]) / 4

                J = dN_dxi.T @ coords
                dN_dx = dN_dxi @ np.linalg.inv(J)

                B = np.zeros((3,8))
                for i in range(4):
                    B[0,2*i]   = dN_dx[i,0]
                    B[1,2*i+1] = dN_dx[i,1]
                    B[2,2*i]   = dN_dx[i,1]
                    B[2,2*i+1] = dN_dx[i,0]

                Ke += B.T @ D @ B * np.linalg.det(J)

        return Ke

class ElementT3:
    def __init__(self, mat):
        self.mat = mat

    def stiffness(self, coords):
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]

        A = 0.5*np.linalg.det([[1,x1,y1],[1,x2,y2],[1,x3,y3]])

        b = [y2-y3, y3-y1, y1-y2]
        c = [x3-x2, x1-x3, x2-x1]

        B = (1/(2*A))*np.array([
            [b[0],0,b[1],0,b[2],0],
            [0,c[0],0,c[1],0,c[2]],
            [c[0],b[0],c[1],b[1],c[2],b[2]]
        ])

        return B.T @ self.mat.D @ B * A * 2
