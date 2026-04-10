import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt

from Mesh import Mesh
from Material import Material
from elements import ElementQ4, ElementT3
from FEM import FEM_Solver, get_radial_stress, plot_all, lame

def run_simulation():
    try:
        # 1. Lấy dữ liệu từ các ô nhập liệu
        E_val = float(entry_E.get())
        nu_val = float(entry_nu.get())
        Ri_val = float(entry_Ri.get())
        Ro_val = float(entry_Ro.get())
        pi_val = float(entry_pi.get())
        po_val = float(entry_po.get())
        nr_val = int(entry_nr.get())
        nt_val = int(entry_nt.get())
    except ValueError:
        messagebox.showerror("Lỗi nhập liệu", "Vui lòng nhập đúng định dạng số (số nguyên hoặc số thập phân)!")
        return

    # Lấy các lựa chọn
    domain = var_domain.get()
    elem_type = var_elem.get()
    is_compare = var_compare.get()

    # Khởi tạo vật liệu
    mat = Material(E_val, nu_val)

    if is_compare:
        # CHẾ ĐỘ SO SÁNH: Chạy cả Q4 và T3 để vẽ biểu đồ so sánh ứng suất
        mesh_q4 = Mesh(Ri_val, Ro_val, nr_val, nt_val, "Q4", mode=domain)
        mesh_t3 = Mesh(Ri_val, Ro_val, nr_val, nt_val, "T3", mode=domain)

        # Giải Q4
        fem_q4 = FEM_Solver(mesh_q4, ElementQ4(mat))
        fem_q4.apply_pressure(Ri_val, pi_val)
        fem_q4.solve()

        # Giải T3
        fem_t3 = FEM_Solver(mesh_t3, ElementT3(mat))
        fem_t3.apply_pressure(Ri_val, pi_val)
        fem_t3.solve()

        # Giải tích (Exact solution)
        r = np.linspace(Ri_val, Ro_val, 200)
        sr_exact, st_exact = lame(r, Ri_val, Ro_val, pi_val, po_val)

        # Trích xuất ứng suất FEM
        r_q4, sr_q4, st_q4 = get_radial_stress(mesh_q4, fem_q4.element, fem_q4.U)
        r_t3, sr_t3, st_t3 = get_radial_stress(mesh_t3, fem_t3.element, fem_t3.U)

        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(r, sr_exact, 'k-', label="Calculus")
        plt.scatter(r_q4, sr_q4, s=15, label=f"FEM ({domain}) - Q4")
        plt.scatter(r_t3, sr_t3, s=15, marker='x', label=f"FEM ({domain}) - T3")
        plt.xlabel("r (m)"); plt.ylabel("Ứng suất (Pa)")
        plt.legend(); plt.title("Ứng suất hướng kính (\u03C3_r)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(r, st_exact, 'k-', label="Calculus (Exact)")
        plt.scatter(r_q4, st_q4, s=15, label=f"FEM ({domain}) - Q4")
        plt.scatter(r_t3, st_t3, s=15, marker='x', label=f"FEM ({domain}) - T3")
        plt.xlabel("Bán kính r (m)"); plt.ylabel("Ứng suất (Pa)")
        plt.legend(); plt.title("Ứng suất tiếp (\u03C3_\u03B8)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    else:
        # CHẾ ĐỘ ĐƠN LẺ: Chạy 1 loại phần tử và xuất ra hình ảnh Mesh + Biến dạng + Trường ứng suất
        mesh = Mesh(Ri_val, Ro_val, nr_val, nt_val, elem_type, mode=domain)
        ElementClass = ElementQ4 if elem_type == "Q4" else ElementT3
        
        fem = FEM_Solver(mesh, ElementClass(mat))
        fem.apply_pressure(Ri_val, pi_val)
        fem.solve()

        # Gọi hàm plot_all đã có sẵn trong file FEM.py của bạn
        plot_all(mesh, fem.U, fem.element, title=f"Kết quả FEM: Mô hình {domain} - Phần tử {elem_type}")

# ==========================================
# THIẾT KẾ GIAO DIỆN (GUI)
# ==========================================
root = tk.Tk()
root.title("piping presure")
root.geometry("450x500")
root.resizable(False, False)

# Style
style = ttk.Style()
style.theme_use('clam')

# Khung nhập thông số
frame_inputs = ttk.LabelFrame(root, text="Thông số đầu vào", padding=(10, 10))
frame_inputs.pack(fill="x", padx=10, pady=5)

def create_input(frame, label_text, default_val, row):
    ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2, padx=5)
    entry = ttk.Entry(frame, width=15)
    entry.insert(0, str(default_val))
    entry.grid(row=row, column=1, pady=2, padx=5)
    return entry

entry_E  = create_input(frame_inputs, "Module đàn hồi E (Pa):", 210e9, 0)
entry_nu = create_input(frame_inputs, "Hệ số Poisson \u03BD:", 0.3, 1)
entry_Ri = create_input(frame_inputs, "Bán kính trong Ri (m):", 0.05, 2)
entry_Ro = create_input(frame_inputs, "Bán kính ngoài Ro (m):", 0.1, 3)
entry_pi = create_input(frame_inputs, "Áp suất trong pi (Pa):", 1e6, 4)
entry_po = create_input(frame_inputs, "Áp suất ngoài po (Pa):", 0, 5)
entry_nr = create_input(frame_inputs, "Số phần tử theo bán kính:", 12, 6)
entry_nt = create_input(frame_inputs, "Số phần tử theo chu vi:", 20, 7)

# Khung tùy chọn
frame_options = ttk.LabelFrame(root, text="Tùy chọn cấu hình", padding=(10, 10))
frame_options.pack(fill="x", padx=10, pady=5)

# Lựa chọn Mô hình (1/4 hay Full)
var_domain = tk.StringVar(value="quarter")
ttk.Label(frame_options, text="Mô hình:").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(frame_options, text="1/4 circle", variable=var_domain, value="quarter").grid(row=0, column=1, sticky="w")
ttk.Radiobutton(frame_options, text="full circle", variable=var_domain, value="full").grid(row=0, column=2, sticky="w")

# Lựa chọn Loại phần tử (Tam giác hay Tứ giác)
var_elem = tk.StringVar(value="Q4")
ttk.Label(frame_options, text="Phần tử:").grid(row=1, column=0, sticky="w", pady=5)
ttk.Radiobutton(frame_options, text="Tứ giác (Q4)", variable=var_elem, value="Q4").grid(row=1, column=1, sticky="w")
ttk.Radiobutton(frame_options, text="Tam giác (T3)", variable=var_elem, value="T3").grid(row=1, column=2, sticky="w")

# Chế độ so sánh
var_compare = tk.BooleanVar(value=False)
chk_compare = ttk.Checkbutton(frame_options, text="so sánh Q4 và T3 với kết quả giải tích", variable=var_compare)
chk_compare.grid(row=2, column=0, columnspan=3, sticky="w", pady=10)

# Nút thực thi
btn_run = ttk.Button(root, text="Simulation", command=run_simulation)
btn_run.pack(pady=15, ipadx=10, ipady=5)

root.mainloop()