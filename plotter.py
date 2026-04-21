import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot_deformation(mesh_obj, U, scale=100):
    fig, ax = plt.subplots(figsize=(8, 6))
    for e in mesh_obj.elements:
        coords = mesh_obj.nodes[e]
        # Lưới ban đầu (nét đứt)
        poly = Polygon(coords, closed=True, fill=False, edgecolor='gray', linestyle='--', alpha=0.4)
        ax.add_patch(poly)
        # Lưới biến dạng
        def_coords = []
        for n in e:
            def_coords.append([
                mesh_obj.nodes[n][0] + U[2*n] * scale,
                mesh_obj.nodes[n][1] + U[2*n+1] * scale
            ])
        poly_def = Polygon(def_coords, closed=True, fill=False, edgecolor='red', linewidth=1.2)
        ax.add_patch(poly_def)
    
    ax.set_title(f"Biến dạng (Phóng đại {scale}x)")
    ax.set_aspect('equal')
    ax.autoscale()
    ax.grid(True, linestyle=':', alpha=0.5)
    return fig

def plot_contour(mesh_obj, values, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    patches = []
    for e in mesh_obj.elements:
        patches.append(Polygon(mesh_obj.nodes[e], closed=True))
    
    p = PatchCollection(patches, cmap='jet')
    p.set_array(np.array(values))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax, label="Giá trị")
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.autoscale()
    return fig

# Hàm vẽ hội tụ giữ nguyên từ turn trước
def plot_convergence(N_list, err_Q4_ur, err_T3_ur, err_Q4_st, err_T3_st):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(N_list, err_Q4_ur, 'b-o', mfc='none', label='Q4')
    ax1.plot(N_list, err_T3_ur, 'r-s', mfc='none', label='T3')
    ax1.set_title('Hội tụ Chuyển vị $u_r$')
    ax1.set_xlabel('Số phần tử (nr)'); ax1.set_ylabel('Sai số (%)'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(N_list, err_Q4_st, 'b-o', mfc='none', label='Q4')
    ax2.plot(N_list, err_T3_st, 'r-s', mfc='none', label='T3')
    ax2.set_title('Hội tụ Ứng suất $\sigma_\\theta$')
    ax2.set_xlabel('Số phần tử (nr)'); ax2.set_ylabel('Sai số (%)'); ax2.legend(); ax2.grid(True)
    return fig

# Thêm vào plotter.py
def plot_stress_all(mesh_obj, cart_s, polar_s):
    # Tạo khung gồm 2 hàng, 3 cột
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # cart_s = (sx, sy, sz, txy, vm)
    # polar_s = (r, theta, sr, st)
    
    # Tiêu đề theo đúng thứ tự 3x2 bạn yêu cầu
    titles = [
        r"Ứng suất $\sigma_x$", r"Ứng suất $\sigma_y$", r"Ứng suất $\sigma_z$",
        r"Ứng suất $\tau_{xy}$", r"Ứng suất $\sigma_r$", r"Ứng suất $\sigma_\theta$"
    ]
    
    # Sắp xếp dữ liệu tương ứng với tiêu đề (Lấy đúng index từ hàm get_element_stresses)
    data_list = [
        cart_s[0], cart_s[1], cart_s[2],  # Hàng 1: sx, sy, sz
        cart_s[3], polar_s[2], polar_s[3] # Hàng 2: txy, sr, st
    ]
    
    axes_flat = axes.flatten()
    
    for i in range(6):
        ax = axes_flat[i]
        values = data_list[i]
        
        # Tạo lại các mảnh (patches) cho mỗi subplot
        patches = []
        for e in mesh_obj.elements:
            patches.append(Polygon(mesh_obj.nodes[e], closed=True))
        
        p = PatchCollection(patches, cmap='jet')
        p.set_array(np.array(values))
        ax.add_collection(p)
        
        fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles[i], fontsize=12)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.axis('off') # Ẩn trục tọa độ cho đẹp
        
    return fig

def plot_convergence_ur(n_list, ur_Q4, ur_T3, ur_exact):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(n_list, ur_Q4, '-ob', label='Q4 (Tứ giác)', linewidth=2, markersize=6)
    ax.plot(n_list, ur_T3, '-sr', label='T3 (Tam giác)', linewidth=2, markersize=6)
    ax.axhline(y=ur_exact, color='k', linestyle='--', linewidth=2.5, label='Giải tích (Analytical)')
    
    ax.set_title('Khảo sát hội tụ: Chuyển vị $u_r$ tại biên trong')
    ax.set_xlabel('Số phần tử theo bán kính (n)')
    ax.set_ylabel('$u_r$ (m)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_convergence_simple(n_list, err_Q4, err_T3):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_list, err_Q4, '-ob', label='Q4 (Tứ giác)', linewidth=2)
    ax.plot(n_list, err_T3, '-sr', label='T3 (Tam giác)', linewidth=2)
    
    ax.set_title('Sai số chuyển vị $u_r$ (%)')
    ax.set_xlabel('Số phần tử theo bán kính (n)')
    ax.set_ylabel('Sai số (%)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_comparison_8_plots(results_T3, results_Q4, results_Ana):
    """
    results_T3/Q4: Dictionary chứa (r, values) cho từng loại ứng suất
    results_Ana: Dictionary chứa (r, values) giải tích
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()

    titles = [
        "Chuyển vị $u_r$", "Ứng suất $\sigma_x$", 
        "Ứng suất $\sigma_y$", "Ứng suất $\sigma_z$",
        "Ứng suất $\\tau_{xy}$", "Ứng suất $\sigma_r$", 
        "Ứng suất $\sigma_\\theta$", "Ứng suất Von Mises"
    ]
    
    keys = ['ur', 'sx', 'sy', 'sz', 'txy', 'sr', 'st', 'vm']

    for i, key in enumerate(keys):
        ax = axes[i]
        
        # --- THÊM DÒNG NÀY ---
        # Chọn đúng mảng bán kính: 'r_u' cho chuyển vị (nút), 'r' cho ứng suất (phần tử)
        x_key = 'r_u' if key == 'ur' else 'r'
        
        # Sửa lại 'r' thành x_key ở các dòng vẽ
        # Vẽ T3
        ax.scatter(results_T3[x_key], results_T3[key], color='red', s=15, label='T3', alpha=0.6)
        # Vẽ Q4
        ax.scatter(results_Q4[x_key], results_Q4[key], color='blue', s=15, label='Q4', alpha=0.6)
        
        # Vẽ Giải tích (nếu có)
        if key in results_Ana:
            ax.plot(results_Ana['r'], results_Ana[key], 'k--', linewidth=2, label='Giải tích')

        ax.set_title(titles[i])
        ax.set_xlabel("Bán kính r")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

    return fig
