import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from mesh import mesh
from material import material
from elements import Q4, T3
from FEM import FEM_Solver
from analytic import get_lame_results
from post import PostProcessor

# Nhập các hàm vẽ từ file chart
from chart import plot_all, plot_comparison

class FEM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Piping Pressure FEM Analysis")
        self.root.geometry("450x550")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        # Khung nhập thông số
        frame_inputs = ttk.LabelFrame(self.root, text="Thông số đầu vào", padding=(10, 10))
        frame_inputs.pack(fill="x", padx=10, pady=5)

        self.entry_E  = self.create_input(frame_inputs, "Module đàn hồi E (Pa):", 210e9, 0)
        self.entry_nu = self.create_input(frame_inputs, "Hệ số Poisson \u03BD:", 0.3, 1)
        self.entry_Ri = self.create_input(frame_inputs, "Bán kính trong Ri (m):", 0.05, 2)
        self.entry_Ro = self.create_input(frame_inputs, "Bán kính ngoài Ro (m):", 0.1, 3)
        self.entry_Pi = self.create_input(frame_inputs, "Áp suất trong Pi (Pa):", 1e7, 4)
        self.entry_Po = self.create_input(frame_inputs, "Áp suất ngoài Po (Pa):", 0.0, 5)

        # Khung cấu hình lưới
        frame_mesh = ttk.LabelFrame(self.root, text="Cấu hình lưới", padding=(10, 10))
        frame_mesh.pack(fill="x", padx=10, pady=5)

        self.entry_nr = self.create_input(frame_mesh, "Số phần tử theo bán kính:", 10, 0)
        self.entry_nt = self.create_input(frame_mesh, "Số phần tử theo góc:", 20, 1)

        self.var_domain = tk.StringVar(value="quarter")
        ttk.Label(frame_mesh, text="Miền tính toán:").grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(frame_mesh, text="1/4 Hình tròn", variable=self.var_domain, value="quarter").grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(frame_mesh, text="Toàn bộ (360°)", variable=self.var_domain, value="full").grid(row=3, column=1, sticky="w")

        # Nút bấm
        self.btn_run = ttk.Button(self.root, text="CHẠY MÔ PHỎNG", command=self.run_simulation)
        self.btn_run.pack(pady=20, ipadx=20, ipady=5)

    def create_input(self, parent, label, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky="e", pady=2, padx=5)
        return entry

    def run_simulation(self):
        try:
            # 1. Lấy dữ liệu từ giao diện
            E  = float(self.entry_E.get())
            nu = float(self.entry_nu.get())
            Ri = float(self.entry_Ri.get())
            Ro = float(self.entry_Ro.get())
            Pi = float(self.entry_Pi.get())
            Po = float(self.entry_Po.get())
            nr = int(self.entry_nr.get())
            nt = int(self.entry_nt.get())
            domain = self.var_domain.get()

            mat = material(E, nu)

            # --- GIẢI BẰNG PHẦN TỬ Q4 ---
            mesh_q4 = mesh(Ri, Ro, nr + 1, nt + 1, "Q4", mode=domain)
            fem_q4 = FEM_Solver(mesh_q4, Q4(mat))
            fem_q4.apply_force(Ri, Pi)
            fem_q4.solve()
            
            post_q4 = PostProcessor(mesh_q4, fem_q4.element, fem_q4.U)
            # Lấy chuyển vị tại nút (ur_q4)
            _, _, ur_q4, _ = post_q4.get_displacements()
            # Lấy ứng suất tại tâm phần tử
            cart_q4, polar_q4 = post_q4.get_element_stresses()

            res_q4 = {
                'ur': ur_q4,
                'sx': cart_q4[0], 'sy': cart_q4[1], 'vm': cart_q4[3],
                'sr': polar_q4[2], 'st': polar_q4[3]
            }
            r_nodes_q4 = np.hypot(mesh_q4.nodes[:,0], mesh_q4.nodes[:,1]) # Trục r cho ur
            r_q4 = polar_q4[0] # Trục r cho stress

            # --- GIẢI BẰNG PHẦN TỬ T3 ---
            mesh_t3 = mesh(Ri, Ro, nr + 1, nt + 1, "T3", mode=domain)
            fem_t3 = FEM_Solver(mesh_t3, T3(mat))
            fem_t3.apply_force(Ri, Pi)
            fem_t3.solve()

            post_t3 = PostProcessor(mesh_t3, fem_t3.element, fem_t3.U)
            # Lấy chuyển vị tại nút (ur_t3)
            _, _, ur_t3, _ = post_t3.get_displacements()
            # Lấy ứng suất tại tâm phần tử
            cart_t3, polar_t3 = post_t3.get_element_stresses()

            res_t3 = {
                'ur': ur_t3,
                'sx': cart_t3[0], 'sy': cart_t3[1], 'vm': cart_t3[3],
                'sr': polar_t3[2], 'st': polar_t3[3]
            }
            r_nodes_t3 = np.hypot(mesh_t3.nodes[:,0], mesh_t3.nodes[:,1]) # Trục r cho ur
            r_t3 = polar_t3[0] # Trục r cho stress

            # --- LẤY KẾT QUẢ GIẢI TÍCH (LAME) ---
            r_exact = np.linspace(Ri, Ro, 100)
            res_exact = get_lame_results(r_exact, Ri, Ro, Pi, Po, E, nu)

            # 2. Hiển thị Contour cho Q4 (Ví dụ)
            plot_all(mesh_q4, fem_q4.U, fem_q4.element, title=f"FEM Results (Q4) - Domain: {domain}")

            # 3. Gọi hàm vẽ so sánh (Truyền đủ các trục tọa độ r cho nút và phần tử)
            plot_comparison(r_exact, res_exact, r_q4, res_q4, r_t3, res_t3, domain, r_nodes_q4, r_nodes_t3)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi trong quá trình tính toán:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FEM_GUI(root)
    root.mainloop()