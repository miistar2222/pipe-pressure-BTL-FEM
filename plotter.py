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
    # Vẫn tạo khung gồm 2 hàng, 3 cột (6 ô)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Chỉ giữ lại 5 tiêu đề (bỏ Von Mises)
    titles = [
        "Ứng suất $\sigma_x$", "Ứng suất $\sigma_y$", "Ứng suất $\\tau_{xy}$",
        "Ứng suất $\sigma_r$", "Ứng suất $\sigma_\\theta$"
    ]
    
    # Chỉ giữ lại 5 dữ liệu tương ứng (bỏ cart_s[3] là Von Mises)
    data_list = [
        cart_s[0], cart_s[1], cart_s[2],
        polar_s[2], polar_s[3]
    ]
    
    axes_flat = axes.flatten()
    
    # Vòng lặp chạy 5 lần để vẽ 5 đồ thị
    for i in range(5):
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
        ax.set_title(titles[i])
        ax.set_aspect('equal')
        ax.autoscale()
        ax.axis('off') # Ẩn trục tọa độ cho đẹp
        
    # Ẩn hoàn toàn ô thứ 6 (index 5) trống để đồ thị trông gọn gàng
    axes_flat[5].set_visible(False)
        
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

# Thêm vào cuối file plotter.py
def plot_comparison(r_nT, ur_nT, r_nQ, ur_nQ, r_ana, ur_ana,
                    r_eT, st_T, r_eQ, st_Q, r_ana_s, st_ana):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Đồ thị chuyển vị (u_r) dọc theo bán kính r
    ax1.plot(r_ana, ur_ana, 'k-', linewidth=2.5, label='Giải tích')
    ax1.plot(r_nQ, ur_nQ, 'bo--', mfc='none', markersize=6, label='Q4 FEM')
    ax1.plot(r_nT, ur_nT, 'rs--', mfc='none', markersize=6, label='T3 FEM')
    ax1.set_title('So sánh Chuyển vị hướng kính ($u_r$) dọc theo r', fontsize=12)
    ax1.set_xlabel('Bán kính $r$ (m)')
    ax1.set_ylabel('Chuyển vị $u_r$ (m)')
    ax1.grid(True, linestyle=':')
    ax1.legend()
    
    # 2. Đồ thị ứng suất vòng (sigma_theta) dọc theo bán kính r
    ax2.plot(r_ana_s, st_ana, 'k-', linewidth=2.5, label='Giải tích')
    ax2.plot(r_eQ, st_Q, 'b^', mfc='none', markersize=6, label='Q4 FEM (Tâm phần tử)')
    ax2.plot(r_eT, st_T, 'rv', mfc='none', markersize=6, label='T3 FEM (Tâm phần tử)')
    ax2.set_title('So sánh Ứng suất vòng ($\sigma_\\theta$) dọc theo r', fontsize=12)
    ax2.set_xlabel('Bán kính tâm phần tử $r$ (m)')
    ax2.set_ylabel('Ứng suất $\sigma_\\theta$ (Pa)')
    ax2.grid(True, linestyle=':')
    ax2.legend()
    
    fig.tight_layout()
    return fig