import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from mesh import mesh
from material import material
from elements import Q4, T3
from FEM import FEM_Solver
from analytic import lame
from post import PostProcessor

# Nhập cả 2 hàm vẽ từ file chart
from chart import plot_all, plot_comparison

class FEM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Piping Pressure FEM")
        self.root.geometry("450x500")
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
        self.entry_pi = self.create_input(frame_inputs, "Áp suất trong pi (Pa):", 1e6, 4)
        self.entry_po = self.create_input(frame_inputs, "Áp suất ngoài po (Pa):", 0, 5)
        self.entry_nr = self.create_input(frame_inputs, "Số phần tử theo bán kính:", 6, 6)
        self.entry_nt = self.create_input(frame_inputs, "Số phần tử theo chu vi:", 10, 7)

        # Khung tùy chọn
        frame_options = ttk.LabelFrame(self.root, text="Tùy chọn cấu hình", padding=(10, 10))
        frame_options.pack(fill="x", padx=10, pady=5)

        self.var_domain = tk.StringVar(value="quarter")
        ttk.Label(frame_options, text="Mô hình:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame_options, text="1/4 circle", variable=self.var_domain, value="quarter").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame_options, text="full circle", variable=self.var_domain, value="full").grid(row=0, column=2, sticky="w")

        self.var_elem = tk.StringVar(value="Q4")
        ttk.Label(frame_options, text="Phần tử:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Radiobutton(frame_options, text="Tứ giác (Q4)", variable=self.var_elem, value="Q4").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frame_options, text="Tam giác (T3)", variable=self.var_elem, value="T3").grid(row=1, column=2, sticky="w")

        self.var_compare = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_options, text="So sánh Q4 và T3 với kết quả giải tích", variable=self.var_compare).grid(row=2, column=0, columnspan=3, sticky="w", pady=10)

        # Nút thực thi
        btn_run = ttk.Button(self.root, text="Simulation", command=self.run_simulation)
        btn_run.pack(pady=15, ipadx=10, ipady=5)

    def create_input(self, frame, label_text, default_val, row):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2, padx=5)
        entry = ttk.Entry(frame, width=15)
        entry.insert(0, str(default_val))
        entry.grid(row=row, column=1, pady=2, padx=5)
        return entry

    def run_simulation(self):
        try:
            E_val = float(self.entry_E.get())
            nu_val = float(self.entry_nu.get())
            Ri_val = float(self.entry_Ri.get())
            Ro_val = float(self.entry_Ro.get())
            pi_val = float(self.entry_pi.get())
            po_val = float(self.entry_po.get())
            nr_val = int(self.entry_nr.get())
            nt_val = int(self.entry_nt.get())
        except ValueError:
            messagebox.showerror("Nhập sai ròi, nhập số đi mài")
            return

        domain = self.var_domain.get()
        elem_type = self.var_elem.get()
        is_compare = self.var_compare.get()

        mat = material(E_val, nu_val)

        if is_compare:
            # 1. Tính toán
            mesh_q4 = mesh(Ri_val, Ro_val, nr_val, nt_val, "Q4", mode=domain)
            mesh_t3 = mesh(Ri_val, Ro_val, nr_val, nt_val, "T3", mode=domain)

            fem_q4 = FEM_Solver(mesh_q4, Q4(mat))
            fem_q4.apply_force(Ri_val, pi_val)
            fem_q4.solve()

            fem_t3 = FEM_Solver(mesh_t3, T3(mat))
            fem_t3.apply_force(Ri_val, pi_val)
            fem_t3.solve()

            # 2. Hậu xử lý dữ liệu
            r_exact = np.linspace(Ri_val, Ro_val, 200)
            sr_exact, st_exact = lame(r_exact, Ri_val, Ro_val, pi_val, po_val)

            post_q4 = PostProcessor(mesh_q4, fem_q4.element, fem_q4.U)
            _, polar_q4 = post_q4.get_element_stresses()
            r_q4, _, sr_q4, st_q4 = polar_q4

            post_t3 = PostProcessor(mesh_t3, fem_t3.element, fem_t3.U)
            _, polar_t3 = post_t3.get_element_stresses()
            r_t3, _, sr_t3, st_t3 = polar_t3

            # 3. Giao nhiệm vụ vẽ biểu đồ cho chart.py
            plot_comparison(r_exact, sr_exact, st_exact, r_q4, sr_q4, st_q4, r_t3, sr_t3, st_t3, domain)

        else:
            mesh_obj = mesh(Ri_val, Ro_val, nr_val, nt_val, elem_type, mode=domain)
            ElementClass = Q4 if elem_type == "Q4" else T3
            
            fem = FEM_Solver(mesh_obj, ElementClass(mat))
            fem.apply_force(Ri_val, pi_val)
            fem.solve()

            # Giao nhiệm vụ vẽ biểu đồ cho chart.py
            plot_all(mesh_obj, fem.U, fem.element, title=f"Kết quả FEM: Mô hình {domain} - Phần tử {elem_type}")