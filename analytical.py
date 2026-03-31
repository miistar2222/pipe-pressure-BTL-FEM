import numpy as np

def lame_solution(r, Ri, Ro, Pi, Po, E, nu):
    """
    Tính toán nghiệm giải tích Lame cho ống trụ dày (Biến dạng phẳng).
    
    Tham số:
    r : Bán kính tại điểm cần tính (có thể là mảng numpy)
    Ri, Ro : Bán kính trong và ngoài
    Pi, Po : Áp suất trong và ngoài
    E, nu : Modulus đàn hồi và hệ số Poisson
    
    Trả về:
    sigma_r, sigma_theta, u_r
    """
    # Các hằng số trung gian trong công thức Lame
    denominator = Ro**2 - Ri**2
    term1 = (Pi * Ri**2 - Po * Ro**2) / denominator
    term2 = (Pi - Po) * (Ri**2 * Ro**2) / (r**2 * denominator)
    
    # 1. Ứng suất hướng tâm (Radial stress)
    sigma_r = term1 - term2
    
    # 2. Ứng suất tiếp tuyến (Hoop stress)
    sigma_theta = term1 + term2
    
    # 3. Ứng suất dọc trục (Axial stress) - Cho biến dạng phẳng epsilon_z = 0
    sigma_z = nu * (sigma_r + sigma_theta)
    
    # 4. Chuyển vị hướng tâm (Radial displacement)
    # Công thức: u_r = (1+nu)/E * [ (1-2nu)*term1*r + term2/r ]
    u_r = ((1 + nu) / E) * ((1 - 2 * nu) * term1 * r + term2 / r)
    
    # 5. Ứng suất Von Mises (Plane Strain)
    # sigma_vM = sqrt(0.5 * [(s_r - s_t)^2 + (s_t - s_z)^2 + (s_z - s_r)^2])
    von_mises = np.sqrt(0.5 * (
        (sigma_r - sigma_theta)**2 + 
        (sigma_theta - sigma_z)**2 + 
        (sigma_z - sigma_r)**2
    ))
    
    return sigma_r, sigma_theta, u_r, von_mises

def get_analytical_results(nodes, Ri, Ro, Pi, Po, E, nu):
    """
    Hàm tiện ích để lấy nghiệm giải tích tại tất cả các vị trí nút của lưới FEM.
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    sr, st, ur, vM = lame_solution(r, Ri, Ro, Pi, Po, E, nu)
    
    # Chuyển đổi chuyển vị tọa độ cực sang tọa độ Descartes (u_x, u_y)
    ux = ur * np.cos(theta)
    uy = ur * np.sin(theta)
    
    return ux, uy, sr, st, vM