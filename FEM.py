import numpy as np
import matplotlib.pyplot as plt


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
