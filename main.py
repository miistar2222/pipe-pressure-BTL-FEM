import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

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
        self.root.title("Phần mềm FEM - Trụ rỗng 2D (Plane Strain)")
        self.root.geometry("1400x900")
        self.canvas = None
        self.create_widgets()

    def create_widgets(self):
        # --- Cột bên trái: Nhập liệu ---
        left_frame = tk.Frame(self.root, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Nhóm 1: Thông số đầu vào
        input_box = tk.LabelFrame(left_frame, text="Thông số đầu vào", padx=10, pady=10)
        input_box.pack(fill=tk.X)

        self.inputs = {}
        fields = [
            ("Ri (Bán kính trong)", "10"), 
            ("Ro (Bán kính ngoài)", "20"), 
            ("nr (Số lớp phần tử r)", "12"), 
            ("nt (Số lớp phần tử theta)", "20"), 
            ("E (Modun đàn hồi)", "2e11"), 
            ("nu (Hệ số Poisson)", "0.3"),
            ("pi (Áp suất trong)", "1e8"),
            ("po (Áp suất ngoài)", "0"),
            ("Scale (Phóng đại biến dạng)", "100")
        ]

        for i, (label, default) in enumerate(fields):
            tk.Label(input_box, text=label).grid(row=i, column=0, sticky="w")
            ent = tk.Entry(input_box, width=15)
            ent.insert(0, default)
            ent.grid(row=i, column=1, pady=2)
            # Lưu key theo tên ngắn gọn để dễ lấy data
            key = label.split(" ")[0]
            self.inputs[key] = ent

        # Nhóm 2: Lựa chọn phần tử
        elem_box = tk.LabelFrame(left_frame, text="Loại phần tử", padx=10, pady=5)
        elem_box.pack(fill=tk.X, pady=10)
        self.cb_elem = ttk.Combobox(elem_box, values=["Q4", "T3"], state="readonly")
        self.cb_elem.current(0)
        self.cb_elem.pack()

        # Nhóm 3: Các nút chức năng
        btn_box = tk.LabelFrame(left_frame, text="Điều khiển", padx=10, pady=10)
        btn_box.pack(fill=tk.X)

        tk.Button(btn_box, text="CHUYỂN VỊ BIẾN DẠNG", command=lambda: self.run("disp"), 
                  bg="#e1f5fe", width=20).pack(pady=5)
        
        # Nút Ứng suất - Sẽ show toàn bộ đồ thị
        tk.Button(btn_box, text="ỨNG SUẤT", command=lambda: self.run("stress"), 
                  bg="#fff9c4", width=20).pack(pady=5)
        
        tk.Button(btn_box, text="VON MISES", command=lambda: self.run("vm"), 
                  bg="#f1f8e9", width=20).pack(pady=5)
        
        tk.Button(btn_box, text="KHẢO SÁT HỘI TỤ", command=lambda: self.run("conv"), 
                  bg="#fce4ec", width=20).pack(pady=5)

        tk.Button(btn_box, text="SO SÁNH", command=lambda: self.run("compare"), 
                  bg="#d1c4e9", width=20).pack(pady=5)

        # --- Cột bên phải: Hiển thị đồ thị ---
        self.plot_frame = tk.Frame(self.root, bg="GREY")
        self.plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def get_data(self):
        try:
            return {k: float(v.get()) for k, v in self.inputs.items()}
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập đúng định dạng số")
            return None

    def run(self, mode):
        d = self.get_data()
        if not d: return

        etype = self.cb_elem.get()
        
        # 1. Khởi tạo đối tượng
        mat = material(d['E'], d['nu'])
        elem = Q4(mat) if etype == "Q4" else T3(mat)
        m = mesh(d['Ri'], d['Ro'], int(d['nr']), int(d['nt']), etype)
        
        # 2. Giải bài toán FEM
        solver = FEM_Solver(m, elem)
        solver.assemble()
        solver.apply_force(d['Ri'], d['Ro'], d['pi'], d['po'])
        solver.solve()

        # 3. Hậu xử lý
        ux, uy, ur, ut = get_displacements(m, solver.U)
        cart_s, polar_s = get_element_stresses(m, elem, solver.U)
        
        # Xóa đồ thị cũ
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # 4. Hiển thị dựa trên chế độ chọn
        if mode == "disp":
            fig = plotter.plot_deformation(m, solver.U, d['Scale'])
        
        elif mode == "stress":
            # GỌI HÀM MỚI: Hiển thị đồ thị
            fig = plotter.plot_stress_all(m, cart_s, polar_s)
            
        elif mode == "vm":
            # Hiển thị riêng Von Mises khổ lớn
            fig = plotter.plot_contour(m, cart_s[4], "Ứng suất Von Mises (Pa)")
            
        elif mode == "conv":
            # Logic khảo sát hội tụ
            self.run_convergence_analysis(d)
            return

        elif mode == "compare":
            self.run_compare_analysis(d)
            return

        # Nhúng vào GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def run_convergence_analysis(self, d): 
        # Chế độ khảo sát hội tụ
        n_list = [5, 10, 15, 20]
        err_Q4, err_T3 = [], []

        ana = Analytical(d['Ri'], d['Ro'], d['E'], d['nu'], d['pi'], d['po'])
        ur_exact_val = ana.get_radial_displacement(d['Ri'])

        for n in n_list:
            nt = n * 5                    # tăng nt để cả Q4 và T3 hội tụ mượt

            # ==================== Q4 ====================
            mQ = mesh(d['Ri'], d['Ro'], n, nt, "Q4")
            elemQ = Q4(material(d['E'], d['nu']))
            solQ = FEM_Solver(mQ, elemQ)
            solQ.assemble()
            solQ.apply_force(d['Ri'], d['Ro'], d['pi'], d['po'])
            solQ.solve()

            _, _, urQ_all, _ = get_displacements(mQ, solQ.U)
            idx_in_Q = [i for i, (x, y) in enumerate(mQ.nodes) 
                        if abs(np.hypot(x, y) - d['Ri']) < 1e-5]
            e_Q = np.mean(np.abs(urQ_all[idx_in_Q] - ur_exact_val) / np.abs(ur_exact_val)) * 100
            err_Q4.append(e_Q)

            # ==================== T3 ====================
            mT = mesh(d['Ri'], d['Ro'], n, nt, "T3")
            elemT = T3(material(d['E'], d['nu']))
            solT = FEM_Solver(mT, elemT)
            solT.assemble()
            solT.apply_force(d['Ri'], d['Ro'], d['pi'], d['po'])
            solT.solve()

            _, _, urT_all, _ = get_displacements(mT, solT.U)
            idx_in_T = [i for i, (x, y) in enumerate(mT.nodes) 
                        if abs(np.hypot(x, y) - d['Ri']) < 1e-5]
            e_T = np.mean(np.abs(urT_all[idx_in_T] - ur_exact_val) / np.abs(ur_exact_val)) * 100
            err_T3.append(e_T)

        # Vẽ đồ thị
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        fig = plotter.plot_convergence_simple(n_list, err_Q4, err_T3)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    
    def run_compare_analysis(self, d):
            # --- Lấy dữ liệu T3 ---
            mT = mesh(d['Ri'], d['Ro'], int(d['nr']), int(d['nt']), "T3")
            solT = FEM_Solver(mT, T3(material(d['E'], d['nu'])))
            solT.assemble(); solT.apply_force(d['Ri'], d['Ro'], d['pi'], d['po']); solT.solve()
            _, _, urT, _ = get_displacements(mT, solT.U)
            cart_T, polar_T = get_element_stresses(mT, T3(material(d['E'], d['nu'])), solT.U)
            
            # Tọa độ r cho nút (chuyển vị) và tâm phần tử (ứng suất)
            r_nodes_T = np.hypot(mT.nodes[:, 0], mT.nodes[:, 1])
            r_elem_T = polar_T[0]

            # --- Lấy dữ liệu Q4 ---
            mQ = mesh(d['Ri'], d['Ro'], int(d['nr']), int(d['nt']), "Q4")
            solQ = FEM_Solver(mQ, Q4(material(d['E'], d['nu'])))
            solQ.assemble(); solQ.apply_force(d['Ri'], d['Ro'], d['pi'], d['po']); solQ.solve()
            _, _, urQ, _ = get_displacements(mQ, solQ.U)
            cart_Q, polar_Q = get_element_stresses(mQ, Q4(material(d['E'], d['nu'])), solQ.U)
            
            r_nodes_Q = np.hypot(mQ.nodes[:, 0], mQ.nodes[:, 1])
            r_elem_Q = polar_Q[0]

            # --- Giải tích ---
            ana = Analytical(d['Ri'], d['Ro'], d['E'], d['nu'], d['pi'], d['po'])
            r_arr = np.linspace(d['Ri'], d['Ro'], 100)
            sr_ana, st_ana = ana.get_stresses(r_arr)
            ur_ana = ana.get_radial_displacement(r_arr)

            # --- Đóng gói dữ liệu ---
            res_T3 = {
                'r': r_elem_T, 'sx': cart_T[0], 'sy': cart_T[1], 'sz': cart_T[2], 
                'txy': cart_T[3], 'vm': cart_T[4], 'sr': polar_T[2], 'st': polar_T[3],
                'ur': urT, 'r_u': r_nodes_T # ur vẽ theo nút
            }
            res_Q4 = {
                'r': r_elem_Q, 'sx': cart_Q[0], 'sy': cart_Q[1], 'sz': cart_Q[2], 
                'txy': cart_Q[3], 'vm': cart_Q[4], 'sr': polar_Q[2], 'st': polar_Q[3],
                'ur': urQ, 'r_u': r_nodes_Q
            }
            res_Ana = {'r': r_arr, 'ur': ur_ana, 'sr': sr_ana, 'st': st_ana}

            # Vẽ lại (Lưu ý trong plotter cần sửa nhẹ để ur dùng 'r_u')
            if self.canvas: self.canvas.get_tk_widget().destroy()
            
            # Gọi hàm vẽ (Bạn có thể tùy biến hàm plotter để nhận các dict này)
            fig = plotter.plot_comparison_8_plots(res_T3, res_Q4, res_Ana)
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FEMApp(root)
    root.mainloop()