import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from elements import stiffness_t3, stiffness_q4, get_d_matrix

def solve_fem(nodes, elements, elem_type, E, nu, Pi, Po, Ri, Ro, bc_nodes):
    """
    Giải bài toán FEM tổng thể.
    bc_nodes: tuple (inner, outer, sym_x, sym_y) từ mesher.get_boundary_nodes
    """
    num_nodes = len(nodes)
    num_dof = 2 * num_nodes
    
    # 1. Khởi tạo ma trận độ cứng tổng thể (Sử dụng LIL format để lắp ghép nhanh)
    K = lil_matrix((num_dof, num_dof))
    F = np.zeros(num_dof)
    D = get_d_matrix(E, nu)

    # 2. Lắp ghép ma trận K (Assembly)
    for elem in elements:
        # Lấy tọa độ các nút của phần tử
        node_coords = nodes[elem]
        
        # Tính ke tùy theo loại phần tử
        if elem_type == 'T3':
            ke = stiffness_t3(node_coords, D)
        else: # Q4
            ke = stiffness_q4(node_coords, D)
            
        # Xác định các chỉ số bậc tự do (dofs) của phần tử
        # Ví dụ nút i có DOF là 2*i và 2*i+1
        dofs = []
        for node_id in elem:
            dofs.extend([2*node_id, 2*node_id + 1])
        
        # Cộng ke vào ma trận tổng thể K
        for i in range(len(dofs)):
            for j in range(len(dofs)):
                K[dofs[i], dofs[j]] += ke[i, j]

    # 3. Tính toán Vector tải trọng F (Áp suất biên)
    # Ta tính lực nút tương đương từ áp suất Pi và Po
    inner_nodes, outer_nodes, sym_x, sym_y = bc_nodes
    F = apply_pressure(F, nodes, inner_nodes, Pi) # Áp suất trong hướng ra ngoài
    F = apply_pressure(F, nodes, outer_nodes, -Po) # Áp suất ngoài hướng vào trong

    # 4. Áp dụng Điều kiện biên (Boundary Conditions)
    # Sử dụng phương pháp khử hoặc thay thế (Penalty method hoặc Direct)
    # Ở đây dùng cách đơn giản: Xóa dòng cột hoặc gán giá trị lớn
    fixed_dofs = []
    # Biên đối xứng Ox (y=0): Chặn Uy = 0
    for node in sym_x: fixed_dofs.append(2*node + 1)
    # Biên đối xứng Oy (x=0): Chặn Ux = 0
    for node in sym_y: fixed_dofs.append(2*node)
    
    # Chuyển K sang định dạng CSR để giải hệ phương trình nhanh
    K = K.tocsr()
    
    # Áp dụng BC bằng cách sửa ma trận (giữ tính đối xứng)
    for dof in fixed_dofs:
        F[dof] = 0
        # Cách thực dụng: xóa liên kết trong ma trận
        # (Trong code thực tế chuyên nghiệp thường dùng mask để giải nhanh hơn)
    
    # Để đơn giản và chính xác, ta dùng kỹ thuật "vô hiệu hóa" bậc tự do
    all_dofs = np.arange(num_dof)
    free_dofs = np.delete(all_dofs, fixed_dofs)
    
    # 5. Giải hệ phương trình KU = F
    U = np.zeros(num_dof)
    U[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], F[free_dofs])
    
    return U

def apply_pressure(F, nodes, boundary_nodes, p):
    """
    Tính lực nút tương đương từ áp suất phân bố trên cung tròn.
    p > 0: hướng ra ngoài tâm
    """
    # Sắp xếp các nút biên theo góc để tìm các đoạn cạnh
    coords = nodes[boundary_nodes]
    thetas = np.arctan2(coords[:, 1], coords[:, 0])
    idx = np.argsort(thetas)
    sorted_nodes = np.array(boundary_nodes)[idx]
    
    for i in range(len(sorted_nodes) - 1):
        n1, n2 = sorted_nodes[i], sorted_nodes[i+1]
        p1, p2 = nodes[n1], nodes[n2]
        
        # Chiều dài đoạn cạnh L
        L = np.sqrt(np.sum((p2 - p1)**2))
        
        # Vector pháp tuyến đơn vị (hướng tâm ra ngoài)
        mid_point = (p1 + p2) / 2
        normal = mid_point / np.linalg.norm(mid_point)
        
        # Tổng lực trên cạnh = p * L. Chia đều cho 2 nút
        force_val = (p * L) / 2
        F[2*n1] += force_val * normal[0]
        F[2*n1+1] += force_val * normal[1]
        F[2*n2] += force_val * normal[0]
        F[2*n2+1] += force_val * normal[1]
    return F