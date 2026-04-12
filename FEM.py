import numpy as np

#[K]{U}={F}
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
                dof.extend([2*n, 2*n+1])    #xác định tọa độ nút
            for i in range(len(dof)):
                for j in range(len(dof)):
                    self.K[dof[i], dof[j]] += ke[i, j] #nhồi ma trận nhỏ của từng nút vô ma trận tổng thể
        print(dof)
        print(ke)

    def apply_force(self, Ri, pi):
        #F=pi*L
        nodes_on_Ri = []
        #tìm các nút ở rìa trong
        for i, (x, y) in enumerate(self.mesh.nodes):
            if abs(np.hypot(x, y) - Ri) < 1e-5: #căn(x^2+y^2) = ri thì nó là nút tại viền
                nodes_on_Ri.append(i) 

        # Sắp xếp các nút trên biên trong theo góc theta
        angles = [np.arctan2(self.mesh.nodes[i][1], self.mesh.nodes[i][0]) % (2*np.pi) for i in nodes_on_Ri]
        #angles = [np.arctan2(y, x) % (2*np.pi) for i in nodes_on_Ri]
        sorted_indices = np.argsort(angles)
        nodes_on_Ri = [nodes_on_Ri[i] for i in sorted_indices]

        if self.mesh.mode == "full":    #đóng vòng lặp với full
            nodes_on_Ri.append(nodes_on_Ri[0])

        for i in range(len(nodes_on_Ri)-1): #xét các cặp nút
            n1 = nodes_on_Ri[i]
            n2 = nodes_on_Ri[i+1]
            x1, y1 = self.mesh.nodes[n1]
            x2, y2 = self.mesh.nodes[n2]
            L = np.hypot(x2-x1, y2-y1) #chiều dài giữa 2 nút 
            # Tính vector pháp tuyến hướng ra ngoài
            dx = x2 - x1
            dy = y2 - y1
            alpha = np.arctan2(dy, dx)
            beta = alpha - np.pi / 2
            F_total = pi * L
            F_x = F_total * np.cos(beta)
            F_y = F_total * np.sin(beta)

            # Phân bổ lực áp suất (p*L/2) cho 2 nút của cạnh
            self.F[2*n1]    += F_x / 2      # Lực phương X tại nút n1
            self.F[2*n1+1]  += F_y / 2      # Lực phương Y tại nút n1
            self.F[2*n2]    += F_x / 2      # Lực phương X tại nút n1
            self.F[2*n2+1]  += F_y / 2      # Lực phương X tại nút n1

    def solve(self):
        self.assemble()
        fixed_dofs = []
        
        # Áp đặt điều kiện biên bằng cách:
        # Trục Y: Những nút nằm trên trục dọc sẽ bị chặn không cho chạy ngang (phương x). Chúng chỉ có thể trượt lên hoặc xuống.
        # Trục X: Những nút nằm trên trục ngang sẽ bị chặn không cho chạy đứng (phương y). Chúng chỉ có thể trượt sang trái hoặc phải.
        # Vẫn cho phép dãn nở hướng kính
        for i, (x, y) in enumerate(self.mesh.nodes):
            if abs(x) < 1e-5: fixed_dofs.append(2*i)     # u_x = 0 trên trục Y
            if abs(y) < 1e-5: fixed_dofs.append(2*i+1)   # u_y = 0 trên trục X

            #thay 0 chỗ đkb vào ma trận tổng thể
        for dof in fixed_dofs:
            # Xóa sạch hàng và cột tương ứng với bậc tự do bị chặn
            self.K[dof, :] = 0
            self.K[:, dof] = 0
            
            # Đặt giá trị 1 vào đường chéo
            self.K[dof, dof] = 1
            
            # Đặt lực tại đó bằng 0
            self.F[dof] = 0

        #Giải hệ phương trình
        self.U = np.linalg.solve(self.K, self.F)
