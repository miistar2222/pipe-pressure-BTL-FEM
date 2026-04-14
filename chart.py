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

    # 1. Vẽ lưới ban đầu
    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[0].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    axs[0].set_title("Original Mesh")
    axs[0].set_aspect('equal')

    # 2. Vẽ Vector chuyển vị CHUẨN HÓA, CHẠM ĐÚNG VỊ TRÍ
    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    
    r_nodes = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    Ri = np.min(r_nodes)
    Ro = np.max(r_nodes)
    
    # Tách riêng index của viền trong và viền ngoài
    tol = 1e-5
    inner_idx = np.where(np.abs(r_nodes - Ri) < tol)[0]
    outer_idx = np.where(np.abs(r_nodes - Ro) < tol)[0]
    
    # Quy định chiều dài cố định cho tất cả các mũi tên (bằng 25% bề dày ống)
    arrow_len = (Ro - Ri) * 0.25 
    
    # --- XỬ LÝ VIỀN TRONG ---
    x_in = mesh.nodes[inner_idx, 0]
    y_in = mesh.nodes[inner_idx, 1]
    # Tính độ lớn vector thực tế (cộng thêm 1e-12 để tránh lỗi chia cho 0)
    mag_in = np.hypot(ux[inner_idx], uy[inner_idx]) + 1e-12 
    # Chuẩn hóa để độ dài vector luôn bằng `arrow_len`
    dx_in = (ux[inner_idx] / mag_in) * arrow_len
    dy_in = (uy[inner_idx] / mag_in) * arrow_len
    
    color_ri='red', color_ro='red'
    width=0.01,headwidth=3,headlength=5    

    # pivot='tail': đuôi mũi tên chạm vào nút tọa độ
    axs[1].quiver(x_in, y_in, dx_in, dy_in, color=color_ri, angles='xy', scale_units='xy', scale=1, 
                  width=width, headwidth=headwidth, headlength=headlength, pivot='tail', zorder=5)

    # --- XỬ LÝ VIỀN NGOÀI ---
    x_out = mesh.nodes[outer_idx, 0]
    y_out = mesh.nodes[outer_idx, 1]
    mag_out = np.hypot(ux[outer_idx], uy[outer_idx]) + 1e-12
    dx_out = (ux[outer_idx] / mag_out) * arrow_len
    dy_out = (uy[outer_idx] / mag_out) * arrow_len
    
    # pivot='tip': đầu mũi tên cắm vào nút tọa độ
    axs[1].quiver(x_out, y_out, dx_out, dy_out, color=color_ro, angles='xy', scale_units='xy', scale=1, 
                  width=width, headwidth=headwidth, headlength=headlength, pivot='tip', zorder=5)
    
    axs[1].set_title("Boundary Displacements (Normalized)")
    axs[1].set_aspect('equal')

    # 3. Các biểu đồ ứng suất
    plot_contour(mesh, axs[2], sx, r"$\sigma_x$ (Cartesian)")
    plot_contour(mesh, axs[3], sy, r"$\sigma_y$ (Cartesian)")
    plot_contour(mesh, axs[4], txy, r"$\tau_{xy}$ (Shear)")
    plot_contour(mesh, axs[5], vm, r"Von Mises Stress")
    plot_contour(mesh, axs[6], sr, r"$\sigma_r$ (Radial Stress)")
    plot_contour(mesh, axs[7], st, r"$\sigma_{\theta}$ (Hoop Stress)")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Hàm vẽ biểu đồ so sánh kết quả
def plot_comparison(r_exact, sr_exact, st_exact, r_q4, sr_q4, st_q4, r_t3, sr_t3, st_t3, domain):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_exact, sr_exact, 'k-', label="Calculus")
    plt.scatter(r_q4, sr_q4, s=15, label=f"FEM ({domain}) - Q4")
    plt.scatter(r_t3, sr_t3, s=15, marker='x', label=f"FEM ({domain}) - T3")
    plt.xlabel("r (m)"); plt.ylabel("Ứng suất (Pa)")
    plt.legend(); plt.title("Ứng suất hướng kính (\u03C3_r)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(r_exact, st_exact, 'k-', label="Calculus (Exact)")
    plt.scatter(r_q4, st_q4, s=15, label=f"FEM ({domain}) - Q4")
    plt.scatter(r_t3, st_t3, s=15, marker='x', label=f"FEM ({domain}) - T3")
    plt.xlabel("Bán kính r (m)"); plt.ylabel("Ứng suất (Pa)")
    plt.legend(); plt.title("Ứng suất tiếp (\u03C3_\u03B8)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()