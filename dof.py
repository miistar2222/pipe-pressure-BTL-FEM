import numpy as np

class DOFHandler:
    def __init__(self, nodes):
        self.nodes = nodes
        self.ndof = len(nodes) * 2

    def get_dofs(self):
        """Xác định danh sách bậc tự do bị chặn và tự do"""
        fixed_dofs = []
        for i, (x, y) in enumerate(self.nodes):
            # Trục Y (x = 0): chặn chuyển vị ngang (phương X) -> DOF chẵn
            if np.isclose(x, 0, atol=1e-5):
                fixed_dofs.append(2 * i)
            
            # Trục X (y = 0): chặn chuyển vị đứng (phương Y) -> DOF lẻ
            if np.isclose(y, 0, atol=1e-5):
                fixed_dofs.append(2 * i + 1)
                
        # Loại bỏ trùng lặp nếu có (ở gốc tọa độ) và sắp xếp
        fixed_dofs = sorted(list(set(fixed_dofs)))
        
        # Các bậc tự do không bị chặn (tự do dịch chuyển)
        all_dofs = np.arange(self.ndof)
        free_dofs = np.delete(all_dofs, fixed_dofs)
        
        return fixed_dofs, free_dofs

    def eliminate(self, K_global, F_global):
        """Loại bỏ các hàng và cột tương ứng với bậc tự do bị chặn"""
        fixed_dofs, free_dofs = self.get_dofs()

        # Dùng np.ix_ để trích xuất ma trận con từ các index tự do
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        
        # Trích xuất vector tải con
        F_reduced = F_global[free_dofs]

        return K_reduced, F_reduced, free_dofs

    def reconstruct_full_vector(self, U_reduced, free_dofs):
        """Lắp ghép lại vector chuyển vị đầy đủ sau khi giải xong hệ thu gọn"""
        U_full = np.zeros(self.ndof)
        U_full[free_dofs] = U_reduced
        return U_full