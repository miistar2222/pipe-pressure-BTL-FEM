import numpy as np

class VonMises:
    def __init__(self, stresses_xy, nu):
        """
        Nhận mảng ứng suất xy và hệ số Poisson để tính ứng suất Von Mises.
        - stresses_xy: Mảng 2D numpy chứa [sigma_x, sigma_y, tau_xy] của từng phần tử.
        - nu: Hệ số Poisson (để tính sigma_z trong biến dạng phẳng).
        """
        self.stresses_xy = stresses_xy
        self.nu = nu

    def calculate(self):
        """
        Tính ứng suất tương đương Von Mises cho bài toán biến dạng phẳng.
        Return: Mảng 1D chứa giá trị Von Mises của từng phần tử.
        """
        # Trích xuất các thành phần ứng suất
        sig_x  = self.stresses_xy[:, 0]
        sig_y  = self.stresses_xy[:, 1]
        tau_xy = self.stresses_xy[:, 2]
        
        # Ứng suất theo trục Z (Do điều kiện biến dạng phẳng)
        sig_z = self.nu * (sig_x + sig_y)
        
        # Áp dụng công thức Von Mises
        term1 = (sig_x - sig_y)**2
        term2 = (sig_y - sig_z)**2
        term3 = (sig_z - sig_x)**2
        term4 = 6 * (tau_xy**2)
        
        sig_vm = np.sqrt(0.5 * (term1 + term2 + term3 + term4))
        
        return sig_vm