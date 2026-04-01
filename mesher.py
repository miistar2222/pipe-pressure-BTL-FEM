import numpy as np

def generate_annulus_mesh_q4(Ri, Ro, nr, nt, theta_max=np.pi/2):
    """
    Tạo lưới Q4 cho 1/4 hình vành khăn.
    Ri, Ro: Bán kính trong và ngoài.
    nr: Số phần tử theo phương bán kính.
    nt: Số phần tử theo phương tiếp tuyến (góc).
    """
    
    # 1. Tạo tọa độ các nút (Nodes)
    r = np.linspace(Ri, Ro, nr + 1)
    theta = np.linspace(0, theta_max, nt + 1)
    
    nodes = []
    for t in theta:
        for radius in r:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            nodes.append([x, y])
    nodes = np.array(nodes)

    # 2. Tạo bảng liên kết phần tử (Elements Connectivity)
    elements_q4 = []
    for j in range(nt):
        for i in range(nr):
            # Chỉ số của 4 nút trong 1 ô tứ giác (ngược chiều kim đồng hồ)
            n1 = j * (nr + 1) + i
            n2 = n1 + 1
            n3 = (j + 1) * (nr + 1) + i + 1
            n4 = (j + 1) * (nr + 1) + i
            elements_q4.append([n1, n2, n3, n4])
            
    return nodes, np.array(elements_q4)

def convert_q4_to_t3(elements_q4):
    """
    Chia mỗi phần tử Q4 thành 2 phần tử T3 để so sánh.
    """
    elements_t3 = []
    for e in elements_q4:
        # Chia tứ giác (n1, n2, n3, n4) thành 2 tam giác: (n1, n2, n3) và (n1, n3, n4)
        elements_t3.append([e[0], e[1], e[2]])
        elements_t3.append([e[0], e[2], e[3]])
    return np.array(elements_t3)

def get_boundary_nodes(nodes, Ri, Ro):
    """
    Xác định các nhóm nút biên để áp dụng điều kiện biên và lực.
    """
    inner_nodes = [] # Nút nằm trên Ri (để áp suất Pi)
    outer_nodes = [] # Nút nằm trên Ro (để áp suất Po)
    sym_x_nodes = [] # Nút nằm trên trục x (y=0, chặn Uy)
    sym_y_nodes = [] # Nút nằm trên trục y (x=0, chặn Ux)

    eps = 1e-9 # Dung sai số thực
    for i, (x, y) in enumerate(nodes):
        r = np.sqrt(x**2 + y**2)
        if abs(r - Ri) < eps: inner_nodes.append(i)
        if abs(r - Ro) < eps: outer_nodes.append(i)
        if abs(y) < eps: sym_x_nodes.append(i)
        if abs(x) < eps: sym_y_nodes.append(i)
            
    return inner_nodes, outer_nodes, sym_x_nodes, sym_y_nodes