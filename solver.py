import numpy as np
import matplotlib.pyplot as plt

# =========================
# SOLVER
# =========================
class FEM:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.K = np.zeros((mesh.ndof, mesh.ndof))
        self.F = np.zeros(mesh.ndof)
        self.U = np.zeros(mesh.ndof)

    def assemble(self):
        for e in self.mesh.elements:
            coords = self.mesh.nodes[e]
            Ke = self.element.stiffness(coords)

            dof = []
            for n in e:
                dof += [2*n,2*n+1]

            for i in range(len(dof)):
                for j in range(len(dof)):
                    self.K[dof[i],dof[j]] += Ke[i,j]

    def apply_pressure(self, Ri, pi):
        tol=1e-5
        for e in self.mesh.elements:
            for i in range(len(e)):
                n1 = e[i]
                n2 = e[(i+1)%len(e)]

                x1,y1 = self.mesh.nodes[n1]
                x2,y2 = self.mesh.nodes[n2]

                r1 = np.hypot(x1,y1)
                r2 = np.hypot(x2,y2)

                if abs(r1-Ri)<tol and abs(r2-Ri)<tol:
                    L = np.hypot(x2-x1,y2-y1)
                    xm,ym = (x1+x2)/2,(y1+y2)/2
                    nx,ny = xm/np.hypot(xm,ym), ym/np.hypot(xm,ym)

                    for n in [n1,n2]:
                        self.F[2*n]   += -pi*nx*L/2
                        self.F[2*n+1] += -pi*ny*L/2

    def apply_bc(self):
        fixed=[]
        tol=1e-6

        if self.mesh.mode == "quarter":
            # symmetry BC
            for i,(x,y) in enumerate(self.mesh.nodes):
                if abs(y)<tol: fixed.append(2*i+1)  # uy=0
                if abs(x)<tol: fixed.append(2*i)    # ux=0

        elif self.mesh.mode == "full":
            # rigid body only
            fixed += [0,1]   # fix node 0
            fixed += [3]     # fix ux node 1

        free=list(set(range(self.mesh.ndof))-set(fixed))
        self.U[free]=np.linalg.solve(self.K[np.ix_(free,free)], self.F[free])

    def solve(self):
        self.assemble()
        self.apply_bc()
