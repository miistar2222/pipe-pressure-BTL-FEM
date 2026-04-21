import numpy as np

class FEM_Solver:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.ndof = len(mesh.nodes) * 2 
        self.K = np.zeros((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof)
        self.U = np.zeros(self.ndof)

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

    def apply_force(self, Ri, Ro, pi, po): 
        # Hàm nội bộ xử lý phân bổ lực cho viền bất kỳ
        def apply_pressure_to_boundary(R, p):
            if p == 0: return # Bỏ qua nếu áp suất = 0
            
            nodes_on_boundary = []
            for i, (x, y) in enumerate(self.mesh.nodes):
                if abs(np.hypot(x, y) - R) < 1e-5:
                    nodes_on_boundary.append(i) 

            # Sắp xếp nút theo góc theta
            angles = [np.arctan2(self.mesh.nodes[i][1], self.mesh.nodes[i][0]) % (2*np.pi) for i in nodes_on_boundary]
            sorted_indices = np.argsort(angles)
            nodes_on_boundary = [nodes_on_boundary[i] for i in sorted_indices]

            for i in range(len(nodes_on_boundary)-1):
                n1 = nodes_on_boundary[i]
                n2 = nodes_on_boundary[i+1]
                x1, y1 = self.mesh.nodes[n1]
                x2, y2 = self.mesh.nodes[n2]
                L = np.hypot(x2-x1, y2-y1) 
                
                dx = x2 - x1
                dy = y2 - y1
                alpha = np.arctan2(dy, dx)
                beta = alpha - np.pi / 2
                F_total = p * L
                F_x = F_total * np.cos(beta)
                F_y = F_total * np.sin(beta)

                # Cộng dồn lực (+=)
                self.F[2*n1]    += F_x / 2
                self.F[2*n1+1]  += F_y / 2
                self.F[2*n2]    += F_x / 2
                self.F[2*n2+1]  += F_y / 2

        # Gọi hàm cho mặt trong (đẩy ra) và mặt ngoài (ép vào)
        apply_pressure_to_boundary(Ri, pi)
        apply_pressure_to_boundary(Ro, -po)

    def solve(self):
        #self.assemble()
        fixed_dofs = []
        
        for i, (x, y) in enumerate(self.mesh.nodes):
            if abs(x) < 1e-5: fixed_dofs.append(2*i)     # chặn u_x trên trục Y
            if abs(y) < 1e-5: fixed_dofs.append(2*i+1)   # chặn u_y trên trục X

        for dof in fixed_dofs:
            self.K[dof, :] = 0
            self.K[:, dof] = 0
            self.K[dof, dof] = 1
            self.F[dof] = 0

        self.U = np.linalg.solve(self.K, self.F)