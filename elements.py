import numpy as np

def get_d_matrix(E, nu):
    #Trả về ma trận đàn hồi D cho bài toán Biến dạng phẳng (Plane Strain)
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = factor * np.array([
        [1 - nu,     nu,          0],
        [nu,         1 - nu,      0],
        [0,          0,           (1 - 2 * nu) / 2]
    ])
    return D

# ==========================================
# PHẦN TỬ TAM GIÁC BẬC NHẤT (T3)
# ==========================================

def stiffness_t3(nodes, D):
    """
    Tính ma trận độ cứng ke (6x6) cho phần tử T3
    nodes: Tọa độ 3 nút [[x1, y1], [x2, y2], [x3, y3]]
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    # Tính diện tích tam giác (2 * Area)
    two_area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    area = two_area / 2

    # Các hệ số của ma trận B
    a = [y2 - y3, y3 - y1, y1 - y2]
    b = [x3 - x2, x1 - x3, x2 - x1]

    # Ma trận B (3x6)
    B = np.array([
        [a[0], 0   , a[1], 0   , a[2], 0   ],
        [0   , b[0], 0   , b[1], 0   , b[2]],
        [b[0], a[0], b[1], a[1], b[2], a[2]]
    ]) / two_area

    # ke = B.T * D * B * Area * thickness (giả sử thickness = 1)
    ke = B.T @ D @ B * area
    return ke

# ==========================================
# PHẦN TỬ TỨ GIÁC BẬC NHẤT (Q4)
# ==========================================

def shape_functions_q4(xi, eta):
    """Trả về giá trị 4 hàm dạng tại tọa độ tự nhiên (xi, eta)"""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    # Đạo hàm hàm dạng theo xi và eta
    dN_dxi_eta = 0.25 * np.array([
        [-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)], # dN/dxi
        [-(1 - xi),  -(1 + xi),  (1 + xi),  (1 - xi)]   # dN/deta
    ])
    return N, dN_dxi_eta

def stiffness_q4(nodes, D):
    """
    Tính ma trận độ cứng ke (8x8) cho phần tử Q4 bằng tích phân Gauss 2x2
    nodes: Tọa độ 4 nút [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    ke = np.zeros((8, 8))
    # Điểm Gauss và trọng số cho tích phân 2x2
    gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    weights = [1, 1]

    for i, xi in enumerate(gauss_pts):
        for j, eta in enumerate(gauss_pts):
            N, dN_dxi_eta = shape_functions_q4(xi, eta)
            
            # Ma trận Jacobian J = dN * nodes
            J = dN_dxi_eta @ nodes
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            # Đạo hàm hàm dạng theo hệ tọa độ thực (x, y)
            dN_dx_dy = invJ @ dN_dxi_eta
            
            # Xây dựng ma trận B (3x8)
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k]   = dN_dx_dy[0, k]
                B[1, 2*k+1] = dN_dx_dy[1, k]
                B[2, 2*k]   = dN_dx_dy[1, k]
                B[2, 2*k+1] = dN_dx_dy[0, k]
            
            # Cộng dồn vào ke: B.T * D * B * detJ * weight
            ke += B.T @ D @ B * detJ * weights[i] * weights[j]
            
    return ke