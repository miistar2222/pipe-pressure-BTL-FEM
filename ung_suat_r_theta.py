import numpy as np

class UngSuatRTheta:
    def __init__(self, mesh, stresses_xy):
        self.mesh = mesh
        self.stresses_xy = stresses_xy

    def calculate_polar_stresses(self):
        """
        Chuyển đổi ứng suất (sigma_x, sigma_y, tau_xy) sang tọa độ cực (sigma_r, sigma_theta, tau_r_theta)
        Return: Mảng 2D kích thước (số_phần_tử, 3)
        """
        polar_stresses = []
        
        for i, el in enumerate(self.mesh.elements):
            coords = self.mesh.nodes[el]
            
            # Tìm tọa độ tâm của phần tử để xác định góc theta
            xc = np.mean(coords[:, 0])
            yc = np.mean(coords[:, 1])
            theta = np.arctan2(yc, xc)
            
            c = np.cos(theta)
            s = np.sin(theta)
            
            # Lấy ứng suất x, y đã tính trước đó
            sig_x  = self.stresses_xy[i, 0]
            sig_y  = self.stresses_xy[i, 1]
            tau_xy = self.stresses_xy[i, 2]
            
            # Xoay tensor ứng suất (Vòng tròn Mohr)
            sig_r  = sig_x * c**2 + sig_y * s**2 + 2 * tau_xy * s * c
            sig_t  = sig_x * s**2 + sig_y * c**2 - 2 * tau_xy * s * c
            tau_rt = -(sig_x - sig_y) * s * c + tau_xy * (c**2 - s**2)
            
            polar_stresses.append([sig_r, sig_t, tau_rt])
            
        return np.array(polar_stresses)