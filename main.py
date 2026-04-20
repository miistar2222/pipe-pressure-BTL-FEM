import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Import các class từ các file của bạn
from material import material
from mesh import mesh
from elements import Q4
from dof import DOFHandler
from force import ForceBoundary

def main():
    # ---------------------------------------------------------
    # 1. THIẾT LẬP THÔNG SỐ BÀI TOÁN
    # ---------------------------------------------------------
    Ri, Ro = 10.0, 20.0       # Bán kính trong, ngoài
    nr, nt = 15, 15           # Số phần tử theo phương r và theta
    E, nu = 2e11, 0.3         # Mô đun đàn hồi (Pa) và hệ số Poisson
    P_inner = 1e7             # Áp suất trong (10 MPa)
    
    # Khởi tạo các đối tượng
    mat = material(E, nu)
    m = mesh(Ri, Ro, nr, nt, "Q4", "quarter")
    nodes = m.nodes
    elements = m.elements
    ndof = m.ndof

    # ---------------------------------------------------------
    # 2. LẮP RÁP MA TRẬN ĐỘ CỨNG [K] VÀ VECTOR LỰC [F]
    # ---------------------------------------------------------
    K_global = np.zeros((ndof, ndof))
    elem_q4 = Q4(mat)

    for el in elements:
        coords = nodes[el]
        Ke = elem_q4.stiffness(coords)
        
        # Lấy chỉ số DOF của 4 nút trong phần tử
        dof_map = []
        for n in el:
            dof_map.extend([2*n, 2*n+1])
            
        # Lắp ráp vào ma trận tổng thể
        for i in range(8):
            for j in range(8):
                K_global[dof_map[i], dof_map[j]] += Ke[i, j]

    force_handler = ForceBoundary(nodes, Ri, Ro)
    F_global = force_handler.compute_force_vector(P_inner=P_inner, P_outer=0.0)

    # ---------------------------------------------------------
    # 3. ÁP DỤNG ĐIỀU KIỆN BIÊN & GIẢI HỆ PHƯƠNG TRÌNH
    # ---------------------------------------------------------
    dof_handler = DOFHandler(nodes)
    K_reduced, F_reduced, free_dofs = dof_handler.eliminate(K_global, F_global)
    
    # Giải hệ [K_red] * {U_red} = {F_red}
    U_reduced = np.linalg.solve(K_reduced, F_reduced)
    
    # Lắp ghép lại vector chuyển vị đầy đủ
    U_full = dof_handler.reconstruct_full_vector(U_reduced, free_dofs)

    # ---------------------------------------------------------
    # 4. TÍNH TOÁN CHUYỂN VỊ & ỨNG SUẤT
    # ---------------------------------------------------------
    Ux = U_full[0::2]
    Uy = U_full[1::2]

    # Tính góc theta tại mỗi nút để chuyển đổi hệ tọa độ
    theta_nodes = np.arctan2(nodes[:, 1], nodes[:, 0])

    # Chuyển vị theo phương r và theta
    Ur = Ux * np.cos(theta_nodes) + Uy * np.sin(theta_nodes)
    Utheta = -Ux * np.sin(theta_nodes) + Uy * np.cos(theta_nodes)

    # Tính ứng suất tại các nút (lấy trung bình từ các phần tử lân cận)
    node_stress = np.zeros((len(nodes), 3)) # [sigma_x, sigma_y, tau_xy]
    node_count = np.zeros(len(nodes))

    for el in elements:
        coords = nodes[el]
        # Lấy vector chuyển vị của phần tử
        U_e = np.zeros(8)
        for i, n in enumerate(el):
            U_e[2*i] = Ux[n]
            U_e[2*i+1] = Uy[n]
            
        # Tính ma trận B tại tâm phần tử (xi=0, eta=0) để xấp xỉ
        xi, eta = 0.0, 0.0
        dN_dxi = np.array([
            [-(1-eta), -(1-xi)],
            [ (1-eta), -(1+xi)],
            [ (1+eta),  (1+xi)],
            [-(1+eta),  (1-xi)]
        ]) / 4.0
        J = dN_dxi.T @ coords
        dN_dx = dN_dxi @ np.linalg.inv(J)
        
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
            
        # Ứng suất phần tử = [D][B]{U_e}
        stress_e = mat.D @ B @ U_e
        
        # Cộng dồn ứng suất vào các nút để lấy trung bình
        for n in el:
            node_stress[n] += stress_e
            node_count[n] += 1

    # Trung bình hóa ứng suất tại nút
    for i in range(len(nodes)):
        node_stress[i] /= node_count[i]

    sig_x, sig_y, tau_xy = node_stress[:,0], node_stress[:,1], node_stress[:,2]

    # Chuyển đổi ứng suất sang hệ tọa độ cực r, theta
    sig_r = sig_x * np.cos(theta_nodes)**2 + sig_y * np.sin(theta_nodes)**2 + 2*tau_xy * np.sin(theta_nodes)*np.cos(theta_nodes)
    sig_theta = sig_x * np.sin(theta_nodes)**2 + sig_y * np.cos(theta_nodes)**2 - 2*tau_xy * np.sin(theta_nodes)*np.cos(theta_nodes)

    # ---------------------------------------------------------
    # 5. VẼ ĐỒ THỊ (VISUALIZATION)
    # ---------------------------------------------------------
    # Tạo lưới tam giác từ Q4 để vẽ contour matplotlib
    triangles = []
    for el in elements:
        triangles.append([el[0], el[1], el[2]])
        triangles.append([el[0], el[2], el[3]])
    triang = mtri.Triangulation(nodes[:,0], nodes[:,1], triangles)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Hàm hỗ trợ vẽ contour
    def plot_contour(ax, z_values, title):
        cntr = ax.tricontourf(triang, z_values, levels=20, cmap='jet')
        ax.set_aspect('equal')
        ax.set_title(title)
        fig.colorbar(cntr, ax=ax)

    # Đồ thị 1: Chuyển vị Ux (Phương X)
    plot_contour(axs[0, 0], Ux, 'Chuyển vị $U_x$ (Hệ Descartes)')
    
    # Đồ thị 2: Ứng suất Sigma_x (Phương X)
    plot_contour(axs[0, 1], sig_x, 'Ứng suất $\sigma_x$ (Hệ Descartes)')

    # Đồ thị 3: Chuyển vị Ur (Phương Hướng Tâm)
    plot_contour(axs[1, 0], Ur, 'Chuyển vị hướng tâm $U_r$ (Hệ Tọa Độ Cực)')

    # Đồ thị 4: Ứng suất Sigma_theta (Phương Tiếp Tuyến)
    plot_contour(axs[1, 1], sig_theta, 'Ứng suất tiếp tuyến $\sigma_{\\theta}$ (Hệ Tọa Độ Cực)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()