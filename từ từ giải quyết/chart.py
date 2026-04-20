import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from post import PostProcessor

def plot_contour(mesh, ax, values, title):
    polys = [mesh.nodes[e] for e in mesh.elements]
    pc = PolyCollection(polys, array=values, cmap='jet', edgecolor='none')
    ax.add_collection(pc)
    ax.autoscale()
    plt.colorbar(pc, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')

def plot_all(mesh, U, element, scale=200, title="FEM Results"):
    post = PostProcessor(mesh, element, U)
    ux, uy, _, _ = post.get_displacements()
    cartesian_stress, polar_stress = post.get_element_stresses()
    sx, sy, txy, vm = cartesian_stress
    r, theta, sr, st = polar_stress
    
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    axs = axs.flatten()

    # 1. Vẽ lưới ban đầu (Ô số 0)
    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[0].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    axs[0].set_title("Mesh")
    axs[0].set_aspect('equal')

    # 2. Vẽ LƯỚI BIẾN DẠNG chồng lên LƯỚI GỐC và MŨI TÊN (Ô số 1)
    
    # Bước A: Vẽ lưới gốc làm nền (màu đen, mờ alpha= độ mờ)
    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'k-', linewidth=1, alpha=1)
    
    #chỗ này vẽ nó dell đẹp nên tao để vậy luôn
    '''
    # Bước B: Tính toán và vẽ lưới biến dạng (màu đỏ)
    # Tọa độ mới = Tọa độ cũ + Chuyển vị * Hệ số phóng đại (scale)
    deformed_nodes = mesh.nodes.copy()
    deformed_nodes[:, 0] += ux * scale
    deformed_nodes[:, 1] += uy * scale
    
    for e in mesh.elements:
        pts = deformed_nodes[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'r-', linewidth=0.7)
    '''
    # Bước C: Vẽ Vector chuyển vị chuẩn hóa 
    r_nodes = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    Ri = np.min(r_nodes)
    Ro = np.max(r_nodes)
    
    tol = 1e-5
    inner_idx = np.where(np.abs(r_nodes - Ri) < tol)[0]
    outer_idx = np.where(np.abs(r_nodes - Ro) < tol)[0]
    
    arrow_len = (Ro - Ri) * 0.25 
    color_ri='red'; color_ro='red'
    width=0.01; headwidth=3; headlength=5    

    # Viền trong: pivot='tail' (đuôi chạm nút gốc)
    x_in, y_in = mesh.nodes[inner_idx, 0], mesh.nodes[inner_idx, 1]
    mag_in = np.hypot(ux[inner_idx], uy[inner_idx]) + 1e-12 
    dx_in = (ux[inner_idx] / mag_in) * arrow_len
    dy_in = (uy[inner_idx] / mag_in) * arrow_len
    
    axs[1].quiver(x_in, y_in, dx_in, dy_in, color=color_ri, angles='xy', scale_units='xy', scale=1, 
                  width=width, headwidth=headwidth, headlength=headlength, pivot='tail', zorder=5)

    # Viền ngoài: pivot='tip' (đầu chạm nút gốc)
    x_out, y_out = mesh.nodes[outer_idx, 0], mesh.nodes[outer_idx, 1]
    mag_out = np.hypot(ux[outer_idx], uy[outer_idx]) + 1e-12
    dx_out = (ux[outer_idx] / mag_out) * arrow_len
    dy_out = (uy[outer_idx] / mag_out) * arrow_len
    
    axs[1].quiver(x_out, y_out, dx_out, dy_out, color=color_ro, angles='xy', scale_units='xy', scale=1, 
                  width=width, headwidth=headwidth, headlength=headlength, pivot='tip', zorder=5)
    
    axs[1].set_title(f"Deformed Overlay")
    axs[1].set_aspect('equal')

    # 3. Các biểu đồ ứng suất
    plot_contour(mesh, axs[2], sx, r"$\sigma_x$")
    plot_contour(mesh, axs[3], sy, r"$\sigma_y$")
    plot_contour(mesh, axs[4], txy, r"$\tau_{xy}$")
    plot_contour(mesh, axs[5], vm, r"Von Mises")
    plot_contour(mesh, axs[6], sr, r"$\sigma_r$")
    plot_contour(mesh, axs[7], st, r"$\sigma_{\theta}$")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Hàm vẽ biểu đồ so sánh kết quả
# Thêm r_nodes_q4 và r_nodes_t3 vào tham số đầu vào
def plot_comparison(r_ex, res_ex, r_q4, res_q4, r_t3, res_t3, domain, r_nodes_q4, r_nodes_t3):
    """
    Vẽ bảng so sánh 2x3 cho tất cả các thông số.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Comparison: Analytical vs FEM (Q4 & T3) - Domain: {domain}", fontsize=16)
    
    metrics = [
        ('ur', 'Displacement $u_r$ (m)'),
        ('sx', 'Stress $\sigma_x$ (Pa)'),
        ('sy', 'Stress $\sigma_y$ (Pa)'),
        ('sr', 'Stress $\sigma_r$ (Pa)'),
        ('st', 'Stress $\sigma_{\\theta}$ (Pa)'),
        ('vm', 'Von Mises Stress (Pa)')
    ]
    
    axs = axs.flatten()
    
    for i, (key, label) in enumerate(metrics):
        # Vẽ đường giải tích (chính xác)
        axs[i].plot(r_ex, res_ex[key], 'r-', label="Exact")
        
        # Chọn trục X: Nếu là 'ur' thì dùng bán kính Nút, nếu là ứng suất thì dùng bán kính Phần tử
        x_q4 = r_nodes_q4 if key == 'ur' else r_q4
        axs[i].scatter(x_q4, res_q4[key], color='blue', s=20, alpha=0.6, label="FEM-Q4")
        
        x_t3 = r_nodes_t3 if key == 'ur' else r_t3
        axs[i].scatter(x_t3, res_t3[key], color='green', marker='x', s=20, alpha=0.8, label="FEM-T3")
        
        axs[i].set_title(label)
        axs[i].set_xlabel('Radius (m)')
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    """
    Vẽ bảng so sánh 2x3 cho tất cả các thông số.
    res_ex, res_q4, res_t3 là các dictionary chứa: ur, sx, sy, sr, st, vm
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Comparison: Analytical vs FEM (Q4 & T3) - Domain: {domain}", fontsize=16)
    
    metrics = [
        ('ur', 'Displacement $u_r$ (m)'),
        ('sx', 'Stress $\sigma_x$ (Pa)'),
        ('sy', 'Stress $\sigma_y$ (Pa)'),
        ('sr', 'Stress $\sigma_r$ (Pa)'),
        ('st', 'Stress $\sigma_{\\theta}$ (Pa)'),
        ('vm', 'Von Mises Stress (Pa)')
    ]
    
    axs = axs.flatten()
    
    for i, (key, label) in enumerate(metrics):
        # Vẽ đường giải tích
        axs[i].plot(r_ex, res_ex[key], 'k-', linewidth=2, label="Calculus")
        
        # Vẽ điểm Q4
        axs[i].scatter(r_q4, res_q4[key], color='blue', s=20, alpha=0.6, label="FEM-Q4")
        
        # Vẽ điểm T3
        axs[i].scatter(r_t3, res_t3[key], color='red', marker='x', s=20, alpha=0.6, label="FEM-T3")
        
        axs[i].set_title(label)
        axs[i].set_xlabel("Radius r (m)")
        axs[i].grid(True, linestyle='--', alpha=0.7)
        if i == 0: axs[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()