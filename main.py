import numpy as np
import matplotlib.pyplot as plt
from mesher import generate_mesh
from fem_core import solve_fem
from analytical import lame_solution

# Thông số đầu vào
Ri, Ro = 10, 20
Pi, Po = 100, 0
E, nu = 2e11, 0.3

mesh_levels = [4, 8, 16, 32] # Số phần tử theo phương bán kính
results = {'T3': [], 'Q4': []}

for n in mesh_levels:
    # Bước 1: Chạy cho Q4
    nodes, elems_q4 = generate_mesh(Ri, Ro, n, n*4, type='Q4')
    U_q4 = solve_fem(nodes, elems_q4, ...)
    results['Q4'].append(evaluate_error(U_q4, ...))
    
    # Bước 2: Chạy cho T3
    elems_t3 = convert_to_t3(elems_q4)
    U_t3 = solve_fem(nodes, elems_t3, ...)
    results['T3'].append(evaluate_error(U_t3, ...))

# Bước 3: Vẽ đồ thị hội tụ (Convergence Plot)
plt.loglog(mesh_levels, results['Q4'], label='Q4 Element')
plt.loglog(mesh_levels, results['T3'], label='T3 Element')
plt.xlabel('Mesh Density')
plt.ylabel('Error (%)')
plt.legend()
plt.show()