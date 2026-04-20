import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import các module
from material import material
from elements import Q4, T3
from mesh import mesh
from calculating import FEM_Solver
from disp_and_stress import get_displacements, get_element_stresses
from calculus_FEM import Analytical
import plotter

class FEMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Giao diện giải bài toán FEM 2D - Trụ rỗng")
        self.root.geometry("1100x700")
        self.create_widgets()

    def create_widgets(self):
        input_frame = tk.LabelFrame(self.root, text="Thông số đầu vào", padx=10, pady=10)
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 1. Hình học & Lưới
        tk.Label(input_frame, text="Hình học & Lưới", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=5, sticky="w")
        tk.Label(input_frame, text="Bán kính trong (Ri):").grid(row=1, column=0, sticky="e")
        self.ent_ri = tk.Entry(input_frame, width=10); self.ent_ri.insert(0, "10.0")
        self.ent_ri.grid(row=1, column=1)

        tk.Label(input_frame, text="Bán kính ngoài (Ro):").grid(row=2, column=0, sticky="e")
        self.ent_ro = tk.Entry(input_frame, width=10); self.ent_ro.insert(0, "20.0")
        self.ent_ro.grid(row=2, column=1)

        tk.Label(input_frame, text="Số điểm chia (nr):").grid(row=3, column=0, sticky="e")
        self.ent_nr = tk.Entry(input_frame, width=10); self.ent_nr.insert(0, "10")
        self.ent_nr.grid(row=3, column=1)

        tk.Label(input_frame, text="Số điểm chia (nt):").grid(row=4, column=0, sticky="e")
        self.ent_nt = tk.Entry(input_frame, width=10); self.ent_nt.insert(0, "15")
        self.ent_nt.grid(row=4, column=1)

        # 2. Vật liệu & Tải
        tk.Label(input_frame, text="Vật liệu & Tải", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=4, pady=(15,5), sticky="w")
        tk.Label(input_frame, text="Mô đun E:").grid(row=6, column=0, sticky="e")
        self.ent_e = tk.Entry(input_frame, width=10); self.ent_e.insert(0, "2e11")
        self.ent_e.grid(row=6, column=1)

        tk.Label(input_frame, text="Hệ số Poisson:").grid(row=7, column=0, sticky="e")
        self.ent_nu = tk.Entry(input_frame, width=10); self.ent_nu.insert(0, "0.3")
        self.ent_nu.grid(row=7, column=1)

        tk.Label(input_frame, text="Áp suất (pi):").grid(row=8, column=0, sticky="e")
        self.ent_pi = tk.Entry(input_frame, width=10); self.ent_pi.insert(0, "1e7")
        self.ent_pi.grid(row=8, column=1)

        tk.Label(input_frame, text="Áp suất (po):").grid(row=8, column=2, sticky="e", padx=(10,0))
        self.ent_po = tk.Entry(input_frame, width=10); self.ent_po.insert(0, "0.0")
        self.ent_po.grid(row=8, column=3) 

        # 3. Chế độ
        tk.Label(input_frame, text="Cấu hình mô phỏng", font=("Arial", 10, "bold")).grid(row=9, column=0, columnspan=4, pady=(15,5), sticky="w")
        tk.Label(input_frame, text="Loại phần tử:").grid(row=10, column=0, sticky="e")
        self.cb_element = ttk.Combobox(input_frame, values=["Q4", "T3"], width=7, state="readonly")
        self.cb_element.current(0); self.cb_element.grid(row=10, column=1)

        tk.Label(input_frame, text="Mô hình:").grid(row=11, column=0, sticky="e")
        self.cb_mode = ttk.Combobox(input_frame, values=["quarter", "full"], width=7, state="readonly")
        self.cb_mode.current(0); self.cb_mode.grid(row=11, column=1)

        tk.Label(input_frame, text="Scale biến dạng:").grid(row=12, column=0, sticky="e")
        self.ent_scale = tk.Entry(input_frame, width=10); self.ent_scale.insert(0, "100")
        self.ent_scale.grid(row=12, column=1)

        # CÁC NÚT NHẤN
        self.btn_solve = tk.Button(input_frame, text="Giải & Vẽ Biến Dạng", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), command=lambda: self.run_simulation("deform"))
        self.btn_solve.grid(row=13, column=0, columnspan=4, pady=(20, 5), ipadx=10, ipady=5, sticky="we")

        # NÚT SO SÁNH SAI SỐ
        self.btn_error = tk.Button(input_frame, text="Tính % Sai Số (Chuyển vị & Ứng suất)", bg="#ff9800", fg="white", font=("Arial", 10, "bold"), command=lambda: self.run_simulation("error"))
        self.btn_error.grid(row=14, column=0, columnspan=4, pady=5, ipadx=10, ipady=5, sticky="we")

        self.plot_frame = tk.Frame(self.root, bg="white")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = None

    def run_simulation(self, action="deform"):
        try:
            Ri, Ro = float(self.ent_ri.get()), float(self.ent_ro.get())
            nr, nt = int(self.ent_nr.get()), int(self.ent_nt.get())
            E, nu = float(self.ent_e.get()), float(self.ent_nu.get())
            pi, po = float(self.ent_pi.get()), float(self.ent_po.get())
            elem_type, sim_mode = self.cb_element.get(), self.cb_mode.get()
            scale = float(self.ent_scale.get())

            mat = material(E, nu)
            elem = Q4(mat) if elem_type == "Q4" else T3(mat)
            m = mesh(Ri, Ro, nr, nt, elem_type, sim_mode)
            
            solver = FEM_Solver(m, elem)
            solver.apply_force(Ri, Ro, pi, po)
            solver.solve()

            ux, uy, ur_fem, utheta = get_displacements(m, solver.U)
            cart_stress, polar_stress = get_element_stresses(m, elem, solver.U)

            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            if action == "deform":
                stresses_vm = cart_stress[3]
                fig = plotter.plot_results(m, solver.U, stresses_vm, scale)
            elif action == "error":
                ana_solver = Analytical(Ri, Ro, E, nu, pi, po)
                r_nodes = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
                r_elems, _, sr_fem, st_fem = polar_stress 
                
                # Gọi hàm vẽ % sai số
                fig = plotter.plot_error_comparison(r_nodes, ur_fem, r_elems, sr_fem, st_fem, ana_solver)
                
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Lỗi thực thi", f"Có lỗi: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FEMApp(root)
    root.mainloop()