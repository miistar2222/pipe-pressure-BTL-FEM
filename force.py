import numpy as np

class ForceBoundary:
    def __init__(self, nodes, Ri, Ro):
        self.nodes = nodes
        self.Ri = Ri
        self.Ro = Ro
        self.ndof = len(nodes) * 2

    def compute_force_vector(self, P_inner=0.0, P_outer=0.0):
        F_global = np.zeros(self.ndof)
        
        # 1. Tìm các nút nằm trên biên trong và biên ngoài
        inner_nodes = []
        outer_nodes = []
        
        for i, (x, y) in enumerate(self.nodes):
            r = np.sqrt(x**2 + y**2)
            if np.isclose(r, self.Ri, atol=1e-3):
                inner_nodes.append(i)
            elif np.isclose(r, self.Ro, atol=1e-3):
                outer_nodes.append(i)

        # 2. Đặt lực lên mặt trong (Áp suất đẩy ra ngoài)
        if P_inner != 0.0 and len(inner_nodes) > 1:
            total_f_in = P_inner * (np.pi * self.Ri / 2) # Tổng lực trên 1/4 cung
            f_per_node_in = total_f_in / (len(inner_nodes) - 1)
            
            for i in inner_nodes:
                x, y = self.nodes[i]
                angle = np.arctan2(y, x)
                # Nút ở 2 đầu mút (0 độ và 90 độ) chỉ chịu một nửa diện tích lấy lực
                weight = 0.5 if np.isclose(angle, 0) or np.isclose(angle, np.pi/2) else 1.0
                
                F_global[2*i]   += f_per_node_in * weight * np.cos(angle) # Fx
                F_global[2*i+1] += f_per_node_in * weight * np.sin(angle) # Fy

        # 3. Đặt lực lên mặt ngoài (Áp suất ép vào trong)
        if P_outer != 0.0 and len(outer_nodes) > 1:
            total_f_out = P_outer * (np.pi * self.Ro / 2)
            f_per_node_out = total_f_out / (len(outer_nodes) - 1)
            
            for i in outer_nodes:
                x, y = self.nodes[i]
                angle = np.arctan2(y, x)
                weight = 0.5 if np.isclose(angle, 0) or np.isclose(angle, np.pi/2) else 1.0
                
                # Trừ đi vì lực hướng ngược chiều vector pháp tuyến (ép vào tâm)
                F_global[2*i]   -= f_per_node_out * weight * np.cos(angle) 
                F_global[2*i+1] -= f_per_node_out * weight * np.sin(angle) 
                
        return F_global