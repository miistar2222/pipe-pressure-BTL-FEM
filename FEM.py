import numpy as np

class FEM:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        
        self.ndof = len(mesh.nodes) * 2 
        
        self.K = np.zeros((self.ndof, self.ndof))   #ma trận độ cứng
        self.F = np.zeros(self.ndof)                #vct tải
        self.U = np.zeros(self.ndof)                #vct chuyển vị

    def assemble(self):
        for e in self.mesh.elements:
            coords = self.mesh.nodes[e]
            ke = self.element.stiffness(coords)
            dof = []
            for n in e:
                dof.extend([2*n, 2*n+1])
            for i in range(len(dof)):
                for j in range(len(dof)):
                    self.K[dof[i], dof[j]] += ke[i, j]

    def apply_pressure(self, Ri, pi):
        nodes_on_Ri = []
        for i, (x, y) in enumerate(self.mesh.nodes):
            if abs(np.hypot(x, y) - Ri) < 1e-5:
                nodes_on_Ri.append(i)

        # Sắp xếp các node trên biên trong theo góc theta
        angles = [np.arctan2(self.mesh.nodes[i][1], self.mesh.nodes[i][0]) % (2*np.pi) for i in nodes_on_Ri]
        sorted_indices = np.argsort(angles)
        nodes_on_Ri = [nodes_on_Ri[i] for i in sorted_indices]

        if self.mesh.mode == "full":
            nodes_on_Ri.append(nodes_on_Ri[0])

        for i in range(len(nodes_on_Ri)-1):
            n1 = nodes_on_Ri[i]
            n2 = nodes_on_Ri[i+1]
            x1, y1 = self.mesh.nodes[n1]
            x2, y2 = self.mesh.nodes[n2]
            L = np.hypot(x2-x1, y2-y1)

            # Tính vector pháp tuyến hướng ra ngoài
            nx, ny = y2 - y1, -(x2 - x1)
            L_norm = np.hypot(nx, ny)
            if L_norm > 0:
                nx, ny = nx / L_norm, ny / L_norm

            # Phân bổ lực áp suất (p*L/2) cho 2 node của cạnh
            F_x, F_y = pi * L * nx, pi * L * ny
            self.F[2*n1] += F_x / 2
            self.F[2*n1+1] += F_y / 2
            self.F[2*n2] += F_x / 2
            self.F[2*n2+1] += F_y / 2

    def solve(self):
        self.assemble()
        fixed_dofs = []
        
        # Áp đặt điều kiện biên đối xứng thông minh cho cả "quarter" và "full"
        # Tránh chuyển vị cứng (rigid body motion) nhưng vẫn cho phép dãn nở hướng kính
        for i, (x, y) in enumerate(self.mesh.nodes):
            if abs(x) < 1e-5:
                fixed_dofs.append(2*i)     # u_x = 0 trên trục Y
            if abs(y) < 1e-5:
                fixed_dofs.append(2*i+1)   # u_y = 0 trên trục X

        free_dofs = list(set(range(self.ndof)) - set(fixed_dofs))
        
        K_ff = self.K[np.ix_(free_dofs, free_dofs)]
        F_f = self.F[free_dofs]
        
        # Giải hệ phương trình
        self.U[free_dofs] = np.linalg.solve(K_ff, F_f)

