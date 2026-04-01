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

# =========================
# POST: extract σr σθ
# =========================
def get_radial_stress(mesh, element, U):
    r_list, sr, st = [],[],[]

    for e in mesh.elements:
        coords = mesh.nodes[e]

        dof=[]
        for n in e:
            dof += [2*n,2*n+1]
        u = U[dof]

        # centroid
        xc,yc = np.mean(coords,axis=0)
        r = np.hypot(xc,yc)
        theta = np.arctan2(yc,xc)

        if len(e)==4:
            dN_dxi = np.array([
                [-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]
            ])
            J = dN_dxi.T @ coords
            dN_dx = dN_dxi @ np.linalg.inv(J)

            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i]=dN_dx[i,0]
                B[1,2*i+1]=dN_dx[i,1]
                B[2,2*i]=dN_dx[i,1]
                B[2,2*i+1]=dN_dx[i,0]
        else:
            x1,y1=coords[0];x2,y2=coords[1];x3,y3=coords[2]
            A=0.5*np.linalg.det([[1,x1,y1],[1,x2,y2],[1,x3,y3]])
            b=[y2-y3,y3-y1,y1-y2]
            c=[x3-x2,x1-x3,x2-x1]
            B=(1/(2*A))*np.array([
                [b[0],0,b[1],0,b[2],0],
                [0,c[0],0,c[1],0,c[2]],
                [c[0],b[0],c[1],b[1],c[2],b[2]]
            ])

        stress = element.mat.D @ (B @ u)
        sx,sy,txy = stress

        c=np.cos(theta); s=np.sin(theta)
        sigma_r = sx*c*c + sy*s*s + 2*txy*s*c
        sigma_t = sx*s*s + sy*c*c - 2*txy*s*c

        r_list.append(r)
        sr.append(sigma_r)
        st.append(sigma_t)

    return np.array(r_list), np.array(sr), np.array(st)

def compute_stress(mesh, element, U):
    sx, sy, txy, vm = [], [], [], []

    for e in mesh.elements:
        coords = mesh.nodes[e]

        dof=[]
        for n in e:
            dof += [2*n,2*n+1]
        u = U[dof]

        if len(e)==4:
            dN_dxi = np.array([
                [-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]
            ])
            J = dN_dxi.T @ coords
            dN_dx = dN_dxi @ np.linalg.inv(J)

            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i]=dN_dx[i,0]
                B[1,2*i+1]=dN_dx[i,1]
                B[2,2*i]=dN_dx[i,1]
                B[2,2*i+1]=dN_dx[i,0]
        else:
            x1,y1=coords[0];x2,y2=coords[1];x3,y3=coords[2]
            A=0.5*np.linalg.det([[1,x1,y1],[1,x2,y2],[1,x3,y3]])
            b=[y2-y3,y3-y1,y1-y2]
            c=[x3-x2,x1-x3,x2-x1]
            B=(1/(2*A))*np.array([
                [b[0],0,b[1],0,b[2],0],
                [0,c[0],0,c[1],0,c[2]],
                [c[0],b[0],c[1],b[1],c[2],b[2]]
            ])

        stress = element.mat.D @ (B @ u)
        sx.append(stress[0])
        sy.append(stress[1])
        txy.append(stress[2])

        vm.append(np.sqrt(
            stress[0]**2 - stress[0]*stress[1] +
            stress[1]**2 + 3*stress[2]**2
        ))

    return np.array(sx), np.array(sy), np.array(txy), np.array(vm)

def plot_all(mesh, U, element, scale=200, title="FEM Results"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()

    # =========================
    # 1. Mesh
    # =========================
    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[0].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    axs[0].set_title("Mesh")
    axs[0].set_aspect('equal')

    # =========================
    # 2. Deformation
    # =========================
    deformed = mesh.nodes.copy()
    for i in range(len(mesh.nodes)):
        deformed[i,0] += scale * U[2*i]
        deformed[i,1] += scale * U[2*i+1]

    for e in mesh.elements:
        pts = deformed[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'r-')
    axs[1].set_title("Deformation")
    axs[1].set_aspect('equal')

    # =========================
    # 3. Stress
    # =========================
    sx, sy, txy, vm = compute_stress(mesh, element, U)

    polys = [mesh.nodes[e] for e in mesh.elements]

    def plot_field(ax, values, name):
        pc = PolyCollection(polys, array=values)
        ax.add_collection(pc)
        ax.autoscale()
        fig.colorbar(pc, ax=ax)
        ax.set_title(name)
        ax.set_aspect('equal')

    plot_field(axs[2], sx, "Sigma X")
    plot_field(axs[3], sy, "Sigma Y")
    plot_field(axs[4], txy, "Tau XY")
    plot_field(axs[5], vm, "Von Mises")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# =========================
# ANALYTIC
# =========================
def lame(r,Ri,Ro,pi,po):
    A = (pi*Ri**2 - po*Ro**2)/(Ro**2-Ri**2)
    B = (Ri**2*Ro**2*(pi-po))/(Ro**2-Ri**2)
    return A-B/r**2, A+B/r**2

# =========================
# MAIN
# =========================
E,nu = 210e9,0.3
Ri,Ro = 0.05,0.1
pi,po = 1e6,0

mat = Material(E,nu)

# =========================
# CASE 1: AXISYMMETRIC (1/4)
# =========================
mesh_q_Q4 = Mesh(Ri,Ro,12,20,"Q4",mode="quarter")
mesh_q_T3 = Mesh(Ri,Ro,12,20,"T3",mode="quarter")

fem_q_Q4 = FEM(mesh_q_Q4, ElementQ4(mat))
fem_q_Q4.apply_pressure(Ri,pi)
fem_q_Q4.solve()

fem_q_T3 = FEM(mesh_q_T3, ElementT3(mat))
fem_q_T3.apply_pressure(Ri,pi)
fem_q_T3.solve()

# =========================
# CASE 2: FULL 360
# =========================
mesh_f_Q4 = Mesh(Ri,Ro,12,40,"Q4",mode="full")
mesh_f_T3 = Mesh(Ri,Ro,12,40,"T3",mode="full")

fem_f_Q4 = FEM(mesh_f_Q4, ElementQ4(mat))
fem_f_Q4.apply_pressure(Ri,pi)
fem_f_Q4.solve()

fem_f_T3 = FEM(mesh_f_T3, ElementT3(mat))
fem_f_T3.apply_pressure(Ri,pi)
fem_f_T3.solve()

# analytic
r = np.linspace(Ri,Ro,200)
sr_exact, st_exact = lame(r,Ri,Ro,pi,po)

# FEM extract
r_q_Q4,sr_q_Q4,st_q_Q4 = get_radial_stress(mesh_q_Q4,fem_q_Q4.element,fem_q_Q4.U)
r_f_Q4,sr_f_Q4,st_f_Q4 = get_radial_stress(mesh_f_Q4,fem_f_Q4.element,fem_f_Q4.U)

# =========================
# PLOT FIELD
# =========================
plot_all(mesh_q_Q4, fem_q_Q4.U, fem_q_Q4.element, title="Axisymmetric Q4")
plot_all(mesh_f_Q4, fem_f_Q4.U, fem_f_Q4.element, title="Full 360 Q4")

plot_all(mesh_q_T3, fem_q_T3.U, fem_q_T3.element, title="Axisymmetric T3")
plot_all(mesh_f_T3, fem_f_T3.U, fem_f_T3.element, title="Full 360 T3")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(r, sr_exact, 'k-', label="Exact")
plt.scatter(r_q_Q4, sr_q_Q4, s=10, label="Quarter Q4")
plt.scatter(r_f_Q4, sr_f_Q4, s=10, label="Full Q4")
plt.legend(); plt.title("Sigma r")

plt.subplot(1,2,2)
plt.plot(r, st_exact, 'k-', label="Exact")
plt.scatter(r_q_Q4, st_q_Q4, s=10, label="Quarter Q4")
plt.scatter(r_f_Q4, st_f_Q4, s=10, label="Full Q4")
plt.legend(); plt.title("Sigma theta")

plt.tight_layout()
plt.show()