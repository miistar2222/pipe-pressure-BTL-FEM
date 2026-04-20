import numpy as np

class UngSuatXY:
    def __init__(self, mesh, mat, U_full):
        self.mesh = mesh
        self.mat = mat
        self.U = U_full

    def calculate_stresses(self):
        """
        Tính [sigma_x, sigma_y, tau_xy] tại tâm của mỗi phần tử.
        Return: Mảng 2D có kích thước (số_phần_tử, 3)
        """
        stresses = []
        D = self.mat.D
        
        for el in self.mesh.elements:
            coords = self.mesh.nodes[el]
            
            # Trích xuất vector chuyển vị của riêng phần tử này (Ue)
            Ue = np.zeros(len(el) * 2)
            for i, n in enumerate(el):
                Ue[2*i]     = self.U[2*n]
                Ue[2*i+1]   = self.U[2*n+1]

            # Khởi tạo ma trận B tại tâm phần tử
            if self.mesh.element_type == "Q4":
                # Đạo hàm hàm dạng tại tâm (xi=0, eta=0)
                dN_dxi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / 4.0
                J = dN_dxi.T @ coords
                dN_dx = dN_dxi @ np.linalg.inv(J)
                
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i]   = dN_dx[i, 0]
                    B[1, 2*i+1] = dN_dx[i, 1]
                    B[2, 2*i]   = dN_dx[i, 1]
                    B[2, 2*i+1] = dN_dx[i, 0]
                    
            elif self.mesh.element_type == "T3":
                x1,y1 = coords[0]; x2,y2 = coords[1]; x3,y3 = coords[2]
                A = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                b1 = y2-y3; b2 = y3-y1; b3 = y1-y2
                c1 = x3-x2; c2 = x1-x3; c3 = x2-x1
                
                B = np.array([
                    [b1, 0, b2, 0, b3, 0],
                    [0, c1, 0, c2, 0, c3],
                    [c1, b1, c2, b2, c3, b3]
                ]) / (2*A)

            # Tính ứng suất: sigma = D * B * Ue
            sigma = D @ B @ Ue
            stresses.append(sigma)
            
        return np.array(stresses)