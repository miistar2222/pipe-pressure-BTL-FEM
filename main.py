import numpy as np
import matplotlib.pyplot as plt
from mesher import generate_annulus_mesh_q4, convert_q4_to_t3, get_boundary_nodes
from fem_core import solve_fem
from analytical import lame_solution
from post_processing import calculate_stresses

# --- 1. THIẾT LẬP THÔNG SỐ ---
Ri, Ro = 10, 20
Pi, Po = 100, 0
E, nu = 2e5, 0.3 # N/mm2
levels = [4, 8, 16, 32] # Các mức lưới khảo sát nr
errors_q4, errors_t3 = [], []

plt.figure(figsize=(12, 5))

for nr in levels:
    nt = nr * 2 # Số phần tử theo phương góc
    
    # --- 2. GIẢI Q4 ---
    nodes, elem_q4 = generate_annulus_mesh_q4(Ri, Ro, nr, nt)
    bc = get_boundary_nodes(nodes, Ri, Ro)
    U_q4 = solve_fem(nodes, elem_q4, 'Q4', E, nu, Pi, Po, Ri, Ro, bc)
    res_q4 = calculate_stresses(nodes, elem_q4, U_q4, 'Q4', E, nu)
    
    # --- 3. GIẢI T3 ---
    elem_t3 = convert_q4_to_t3(elem_q4)
    U_t3 = solve_fem(nodes, elem_t3, 'T3', E, nu, Pi, Po, Ri, Ro, bc)
    res_t3 = calculate_stresses(nodes, elem_t3, U_t3, 'T3', E, nu)
    
    # --- 4. TÍNH SAI SỐ (tại nút biên trong Ri) ---
    _, _, ur_ana, _ = lame_solution(Ri, Ri, Ro, Pi, Po, E, nu)
    ur_fem_q4 = np.mean(np.sqrt(U_q4[2*bc[0]]**2 + U_q4[2*bc[0]+1]**2))
    errors_q4.append(abs(ur_fem_q4 - ur_ana)/ur_ana * 100)

# --- 5. VẼ ĐỒ THỊ ---
# Đồ thị hội tụ
plt.subplot(1, 2, 1)
plt.plot(levels, errors_q4, 'o-', label='Q4 Convergence')
plt.xlabel('Number of elements (radial)')
plt.ylabel('Error (%)')
plt.legend()
plt.grid(True)

# Đồ thị so sánh ứng suất (với mức lưới cuối cùng)
r_fine = np.linspace(Ri, Ro, 100)
sr_ana, st_ana, _, vM_ana = lame_solution(r_fine, Ri, Ro, Pi, Po, E, nu)

plt.subplot(1, 2, 2)
plt.plot(r_fine, st_ana, 'k-', label='Analytical Stress_theta')
plt.scatter(res_q4[:, 6], res_q4[:, 5], c='r', marker='x', label='FEM Q4')
plt.xlabel('Radius r')
plt.ylabel('Stress')
plt.legend()
plt.show()