import numpy as np
import matplotlib.pyplot as plt

from Mesh import Mesh
from Material import Material
from elements import ElementQ4, ElementT3
from FEM import FEM_Solver, get_radial_stress, plot_all, lame

# =========================
# MAIN
# =========================
E, nu = 210e9, 0.3
Ri, Ro = 0.05, 0.1
pi, po = 1e6, 0

mat = Material(E, nu)

# =========================
# CASE 1: AXISYMMETRIC (1/4)
# =========================
mesh_q_Q4 = Mesh(Ri, Ro, 12, 20, "Q4", mode="quarter")
mesh_q_T3 = Mesh(Ri, Ro, 12, 20, "T3", mode="quarter")

fem_q_Q4 = FEM_Solver(mesh_q_Q4, ElementQ4(mat))
fem_q_Q4.apply_pressure(Ri, pi)
fem_q_Q4.solve()

fem_q_T3 = FEM_Solver(mesh_q_T3, ElementT3(mat))
fem_q_T3.apply_pressure(Ri, pi)
fem_q_T3.solve()

# =========================
# CASE 2: FULL 360
# =========================
mesh_f_Q4 = Mesh(Ri, Ro, 12, 40, "Q4", mode="full")
mesh_f_T3 = Mesh(Ri, Ro, 12, 40, "T3", mode="full")

fem_f_Q4 = FEM_Solver(mesh_f_Q4, ElementQ4(mat))
fem_f_Q4.apply_pressure(Ri, pi)
fem_f_Q4.solve()

fem_f_T3 = FEM_Solver(mesh_f_T3, ElementT3(mat))
fem_f_T3.apply_pressure(Ri, pi)
fem_f_T3.solve()

# analytic
r = np.linspace(Ri, Ro, 200)
sr_exact, st_exact = lame(r, Ri, Ro, pi, po)

# FEM extract
r_q_Q4, sr_q_Q4, st_q_Q4 = get_radial_stress(mesh_q_Q4, fem_q_Q4.element, fem_q_Q4.U)
r_f_Q4, sr_f_Q4, st_f_Q4 = get_radial_stress(mesh_f_Q4, fem_f_Q4.element, fem_f_Q4.U)

# =========================
# PLOT FIELD
# =========================
plot_all(mesh_q_Q4, fem_q_Q4.U, fem_q_Q4.element, title="Axisymmetric Q4")
plot_all(mesh_f_Q4, fem_f_Q4.U, fem_f_Q4.element, title="Full 360 Q4")

plot_all(mesh_q_T3, fem_q_T3.U, fem_q_T3.element, title="Axisymmetric T3")
plot_all(mesh_f_T3, fem_f_T3.U, fem_f_T3.element, title="Full 360 T3")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(r, sr_exact, 'k-', label="Exact")
plt.scatter(r_q_Q4, sr_q_Q4, s=10, label="Quarter Q4")
plt.scatter(r_f_Q4, sr_f_Q4, s=10, label="Full Q4")
plt.legend(); plt.title("Sigma r")

plt.subplot(1,2,2)
plt.plot(r, st_exact, 'k-', label="Exact")
plt.scatter(r_q_Q4, st_q_Q4, s=10, label="Quarter Q4")
plt.scatter(r_f_Q4, st_f_Q4, s=10, label="Full Q4")
plt.legend(); plt.title("Sigma theta")

plt.tight_layout()
plt.show()