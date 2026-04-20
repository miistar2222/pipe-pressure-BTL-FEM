import numpy as np

class Calculus:
    def __init__(self, Ri, Ro, Pi, E, nu, Po=0.0):
        """
        Khởi tạo các thông số cho bài toán giải tích (Lamé Problem)
        - Ri, Ro: Bán kính trong và ngoài
        - Pi, Po: Áp suất trong và ngoài (Mặc định Po = 0 nếu không có)
        - E, nu: Module đàn hồi và hệ số Poisson
        """
        self.Ri = Ri
        self.Ro = Ro
        self.Pi = Pi
        self.Po = Po
        self.E = E
        self.nu = nu

    def calculate_exact(self, r):
        """
        Tính toán chính xác chuyển vị và ứng suất tại vị trí bán kính r
        r: có thể là 1 giá trị (float) hoặc 1 mảng (numpy array) các bán kính
        """
        # Các hằng số trung gian trong công thức Lamé
        A = (self.Pi * self.Ri**2 - self.Po * self.Ro**2) / (self.Ro**2 - self.Ri**2)
        B = (self.Ri**2 * self.Ro**2 * (self.Pi - self.Po)) / (self.Ro**2 - self.Ri**2)
        
        # 1. Ứng suất hướng kính (Radial stress) và tiếp tuyến (Hoop/Tangential stress)
        sig_r = A - B / (r**2)
        sig_theta = A + B / (r**2)
        
        # Ứng suất dọc trục z do điều kiện biến dạng phẳng (Plane Strain: eps_z = 0)
        sig_z = self.nu * (sig_r + sig_theta)
        
        # 2. Chuyển vị hướng kính (Radial displacement)
        u_r = ((1 + self.nu) / self.E) * (A * (1 - 2 * self.nu) * r + B / r)
        
        # 3. Ứng suất tương đương Von Mises
        term1 = (sig_r - sig_theta)**2
        term2 = (sig_theta - sig_z)**2
        term3 = (sig_z - sig_r)**2
        sig_vm = np.sqrt(0.5 * (term1 + term2 + term3))
        
        return u_r, sig_r, sig_theta, sig_vm