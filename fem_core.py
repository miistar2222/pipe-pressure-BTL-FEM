import numpy as np
from elements import stiffness_t3, stiffness_q4, get_d_matrix

def solve_fem(nodes, elements, elem_type, E, nu, Pi, Po, Ri, Ro, bc_nodes):
    num_nodes = len(nodes)
    num_dof = 2 * num_nodes
    
    # 1. Khởi tạo ma trận K và F bằng NumPy (Dạng đặc - Dense)
    K = np.zeros((num_dof, num_dof))
    F = np.zeros(num_dof)
    D = get_d_matrix(E, nu)

    # 2. Lắp ghép ma trận K
    for elem in elements:
        node_coords = nodes[elem]
        
        # Tính ke (6x6 cho T3, 8x8 cho Q4)
        if elem_type == 'T3':
            ke = stiffness_t3(node_coords, D)
        else:
            ke = stiffness_q4(node_coords, D)
            
        # Xác định chỉ số Bậc tự do (DOF)
        dofs = []
        for node_id in elem:
            dofs.extend([2*node_id, 2*node_id + 1])
        
        # Lắp ghép vào K tổng thể bằng chỉ số mảng của NumPy
        # K[np.ix_(dofs, dofs)] là cách viết tắt để cộng ma trận con vào ma trận lớn
        K[np.ix_(dofs, dofs)] += ke

    # 3. Tính Vector tải trọng F (Áp suất)
    inner_nodes, outer_nodes, sym_x, sym_y = bc_nodes
    F = apply_pressure(F, nodes, inner_nodes, Pi)
    F = apply_pressure(F, nodes, outer_nodes, -Po)

    # 4. Xác định các bậc tự do bị chặn (Fixed DOFs)
    fixed_dofs = []
    for node in sym_x: fixed_dofs.append(2*node + 1) # Chặn Uy tại y=0
    for node in sym_y: fixed_dofs.append(2*node)     # Chặn Ux tại x=0
    
    # Tìm các bậc tự do còn lại (Free DOFs)
    all_dofs = np.arange(num_dof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    
    # 5. Giải hệ phương trình KU = F bằng np.linalg.solve
    # Chúng ta chỉ giải cho các bậc tự do không bị chặn
    U = np.zeros(num_dof)
    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]
    
    U[free_dofs] = np.linalg.solve(K_free, F_free)
    
    return U

def apply_pressure(F, nodes, boundary_nodes, p):
    """Giữ nguyên logic tính lực nút như cũ"""
    coords = nodes[boundary_nodes]
    thetas = np.arctan2(coords[:, 1], coords[:, 0])
    idx = np.argsort(thetas)
    sorted_nodes = np.array(boundary_nodes)[idx]
    
    for i in range(len(sorted_nodes) - 1):
        n1, n2 = sorted_nodes[i], sorted_nodes[i+1]
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        mid_point = (p1 + p2) / 2
        normal = mid_point / np.linalg.norm(mid_point)
        
        force_val = (p * L) / 2
        F[2*n1]   += force_val * normal[0]
        F[2*n1+1] += force_val * normal[1]
        F[2*n2]   += force_val * normal[0]
        F[2*n2+1] += force_val * normal[1]
    return F