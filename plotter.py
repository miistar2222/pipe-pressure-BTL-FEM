import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot_results(mesh_obj, U, stresses_vm, scale=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for e in mesh_obj.elements:
        coords = mesh_obj.nodes[e]
        poly = Polygon(coords, closed=True, fill=False, edgecolor='gray', linestyle='--', alpha=0.6)
        ax1.add_patch(poly)
        
        def_coords = []
        for n in e:
            ux = U[2*n]
            uy = U[2*n+1]
            def_coords.append([
                mesh_obj.nodes[n][0] + ux * scale,
                mesh_obj.nodes[n][1] + uy * scale
            ])
        poly_def = Polygon(def_coords, closed=True, fill=False, edgecolor='red', linewidth=1.2)
        ax1.add_patch(poly_def)
        
    ax1.set_title(f"Lưới ban đầu & Biến dạng (Scale = {scale}x)")
    ax1.autoscale(); ax1.set_aspect('equal'); ax1.grid(True, linestyle=':', alpha=0.6)

    stress_patches = []
    for e in mesh_obj.elements:
        coords = mesh_obj.nodes[e]
        stress_patches.append(Polygon(coords, closed=True))
        
    p = PatchCollection(stress_patches, cmap='jet', alpha=0.9)
    p.set_array(np.array(stresses_vm))
    ax2.add_collection(p)
    
    cbar = fig.colorbar(p, ax=ax2)
    cbar.set_label("Ứng suất Von Mises")
    ax2.set_title("Phân bố ứng suất Von Mises")
    ax2.autoscale(); ax2.set_aspect('equal'); ax2.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    return fig

def plot_error_comparison(r_nodes, ur_fem, r_elems, sr_fem, st_fem, ana_solver):
    """ Vẽ đồ thị CHỈ SO SÁNH % SAI SỐ giữa FEM và Giải tích """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Tính nghiệm giải tích tại đúng các vị trí r của FEM
    ur_ana = ana_solver.get_radial_displacement(r_nodes)
    sr_ana, st_ana = ana_solver.get_stresses(r_elems)
    
    # Hàm tính sai số tuyệt đối % (có bỏ qua lỗi chia 0 nếu giải tích = 0)
    def calc_error(fem_val, ana_val):
        with np.errstate(divide='ignore', invalid='ignore'):
            err = np.abs((fem_val - ana_val) / ana_val) * 100
            err[ana_val == 0] = 0 
        return err

    # 2. Tính % sai số
    err_ur = calc_error(ur_fem, ur_ana)
    err_sr = calc_error(sr_fem, sr_ana)
    err_st = calc_error(st_fem, st_ana)
    
    # 3. Biểu đồ Sai số Chuyển vị
    ax1.plot(r_nodes, err_ur, 'ro', markersize=4, alpha=0.6, label='Sai số $u_r$')
    ax1.set_xlabel('Bán kính r (m)')
    ax1.set_ylabel('Sai số Chuyển vị (%)')
    ax1.set_title('Sai lệch % Chuyển vị hướng kính ($u_r$)')
    ax1.grid(True, linestyle=':')
    ax1.legend()
    
    # 4. Biểu đồ Sai số Ứng suất
    ax2.plot(r_elems, err_sr, 'bo', markersize=4, alpha=0.6, label=r'Sai số $\sigma_r$')
    ax2.plot(r_elems, err_st, 'go', markersize=4, alpha=0.6, label=r'Sai số $\sigma_\theta$')
    ax2.set_xlabel('Bán kính r tâm phần tử (m)')
    ax2.set_ylabel('Sai số Ứng suất (%)')
    ax2.set_title('Sai lệch % Ứng suất hướng kính & vòng')
    ax2.grid(True, linestyle=':')
    ax2.legend()
    
    fig.tight_layout()
    return fig