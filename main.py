import numpy as np
import matplotlib.pyplot as plt
from mesher import generate_annulus_mesh_q4, convert_q4_to_t3, get_boundary_nodes
from fem_core import solve_fem
from analytical import lame_solution
from post_processing import calculate_stresses

# --- 1. THIẾT LẬP THÔNG SỐ HÌNH HỌC & VẬT LIỆU ---
Ri, Ro = 10, 20      # mm
Pi, Po = 100, 0      # MPa
E, nu = 2e5, 0.3     # MPa
levels = [4, 8, 16, 32] # Các mức lưới khảo sát nr

errors_q4 = []
errors_t3 = []

print("Đang bắt đầu tính toán khảo sát hội tụ...")

for nr in levels:
    nt = nr * 4 
    
    # --- A. GIẢI VỚI PHẦN TỬ Q4 ---
    nodes, elem_q4 = generate_annulus_mesh_q4(Ri, Ro, nr, nt)
    # Ép kiểu numpy array cho các chỉ số nút biên
    inner_idx, outer_idx, sx_idx, sy_idx = [np.array(i) for i in get_boundary_nodes(nodes, Ri, Ro)]
    bc_idx = (inner_idx, outer_idx, sx_idx, sy_idx)
    
    U_q4 = solve_fem(nodes, elem_q4, 'Q4', E, nu, Pi, Po, Ri, Ro, bc_idx)
    
    # --- B. GIẢI VỚI PHẦN TỬ T3 ---
    elem_t3 = convert_q4_to_t3(elem_q4)
    U_t3 = solve_fem(nodes, elem_t3, 'T3', E, nu, Pi, Po, Ri, Ro, bc_idx)
    
    # --- C. TÍNH SAI SỐ CHUYỂN VỊ TẠI BIÊN TRONG (Ri) ---
    _, _, ur_ana, _ = lame_solution(Ri, Ri, Ro, Pi, Po, E, nu)
    
    # Q4 Error
    ux_q4 = U_q4[2 * inner_idx]
    uy_q4 = U_q4[2 * inner_idx + 1]
    ur_fem_q4 = np.mean(np.sqrt(ux_q4**2 + uy_q4**2))
    errors_q4.append(abs(ur_fem_q4 - ur_ana) / ur_ana * 100)
    
    # T3 Error
    ux_t3 = U_t3[2 * inner_idx]
    uy_t3 = U_t3[2 * inner_idx + 1]
    ur_fem_t3 = np.mean(np.sqrt(ux_t3**2 + uy_t3**2))
    errors_t3.append(abs(ur_fem_t3 - ur_ana) / ur_ana * 100)
    
    print(f"Hoàn thành nr = {nr}")

# --- 2. VẼ ĐỒ THỊ KHẢO SÁT HỘI TỤ ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(levels, errors_q4, 's-', label='Q4 Element')
plt.plot(levels, errors_t3, 'o--', label='T3 Element')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mật độ lưới (nr)')
plt.ylabel('Sai số chuyển vị (%)')
plt.title('Khảo sát sự hội tụ')
plt.legend()
plt.grid(True, which="both", ls="-")

# --- 3. VẼ MÔ HÌNH LƯỚI (So sánh Q4 và T3) ---
plt.figure(figsize=(12, 5))

# Vẽ lưới Q4
plt.subplot(1, 2, 1)
for e in elem_q4:
    pts = nodes[list(e) + [e[0]]]
    plt.plot(pts[:, 0], pts[:, 1], 'b-', lw=0.5)
plt.axis('equal')
plt.title(f'Mô hình lưới Q4 (nr={nr})')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

# Vẽ lưới T3
plt.subplot(1, 2, 2)
# Sử dụng tripcolor hoặc vẽ từng cạnh của tam giác
for e in elem_t3:
    pts = nodes[list(e) + [e[0]]]
    plt.plot(pts[:, 0], pts[:, 1], 'r-', lw=0.5) # Lưới tam giác vẽ màu đỏ (r-)
plt.axis('equal')
plt.title(f'Mô hình lưới T3 (nr={nr})')
plt.xlabel('x (mm)')

plt.tight_layout()
plt.show()

# --- 4. SO SÁNH ỨNG SUẤT TIẾP TUYẾN ---
res_q4 = calculate_stresses(nodes, elem_q4, U_q4, 'Q4', E, nu)
r_fine = np.linspace(Ri, Ro, 100)
_, st_ana, _, _ = lame_solution(r_fine, Ri, Ro, Pi, Po, E, nu)

plt.subplot(1, 3, 3)
plt.plot(r_fine, st_ana, 'k-', label='Giải tích (Lame)')
plt.scatter(res_q4[:, 6], res_q4[:, 5], c='r', marker='x', s=20, label='FEM Q4')
plt.xlabel('Bán kính r (mm)')
plt.ylabel('Stress_theta (MPa)')
plt.title('So sánh Ứng suất')
plt.legend()
plt.tight_layout()
plt.show()

# --- 5. VẼ BẢN ĐỒ MÀU ỨNG SUẤT VON MISES (SỬA LỖI VALUEERROR) ---
plt.figure(figsize=(8, 6))

# Lấy giá trị vM từ kết quả tính toán của Q4
vM_values = res_q4[:, 3] 

# Vì 1 ô Q4 được chia thành 2 tam giác để vẽ, ta cần lặp lại giá trị màu 2 lần
vM_faces = np.repeat(vM_values, 2)

# Vẽ trường ứng suất
plt.tripcolor(nodes[:, 0], nodes[:, 1], convert_q4_to_t3(elem_q4), 
             facecolors=vM_faces, edgecolors='k', lw=0.1, cmap='jet')

plt.colorbar(label='Von Mises Stress (MPa)')
plt.axis('equal')
plt.title('Phân bố ứng suất Von Mises (Phần tử Q4)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.show()