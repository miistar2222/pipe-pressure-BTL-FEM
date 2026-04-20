import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def _get_triangulation(nodes, elements, elem_type):
    """
    Hàm phụ trợ (ẩn) để tạo lưới tam giác cho matplotlib.
    Matplotlib vẽ contour tốt nhất trên lưới tam giác.
    Nếu là Q4, hàm sẽ tự động chia 1 tứ giác thành 2 tam giác.
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    if elem_type == "T3":
        triangles = elements
    elif elem_type == "Q4":
        triangles = []
        for el in elements:
            # Chia tứ giác (0,1,2,3) thành 2 tam giác (0,1,2) và (0,2,3)
            triangles.append([el[0], el[1], el[2]])
            triangles.append([el[0], el[2], el[3]])
        triangles = np.array(triangles)
    else:
        raise ValueError("Chỉ hỗ trợ element_type là 'T3' hoặc 'Q4'")
        
    return mtri.Triangulation(x, y, triangles)

def plot_mesh(nodes, elements, title="Mô hình lưới FEM"):
    """Xuất đồ thị mô hình lưới (Mesh)"""
    plt.figure(figsize=(6, 6))
    for el in elements:
        polygon = np.append(el, el[0]) # Nối điểm cuối về điểm đầu để tạo vòng khép kín
        x = nodes[polygon, 0]
        y = nodes[polygon, 1]
        plt.plot(x, y, 'k-', lw=0.8)
        
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_deformed_mesh(nodes, elements, U_full, scale=10.0, title="Lưới biến dạng (Deformed Mesh)"):
    """Xuất đồ thị so sánh lưới ban đầu (mờ) và lưới sau biến dạng"""
    plt.figure(figsize=(6, 6))
    
    # 1. Vẽ lưới ban đầu (màu xám nhạt, nét đứt)
    for el in elements:
        polygon = np.append(el, el[0])
        x = nodes[polygon, 0]
        y = nodes[polygon, 1]
        plt.plot(x, y, color='lightgray', linestyle='--', lw=0.8)
        
    # 2. Vẽ lưới biến dạng (màu xanh dương, nét liền)
    # Phóng đại biến dạng lên 'scale' lần để dễ quan sát bằng mắt thường
    nodes_def = np.copy(nodes)
    nodes_def[:, 0] += U_full[0::2] * scale
    nodes_def[:, 1] += U_full[1::2] * scale
    
    for el in elements:
        polygon = np.append(el, el[0])
        x = nodes_def[polygon, 0]
        y = nodes_def[polygon, 1]
        plt.plot(x, y, 'b-', lw=1.2)
        
    plt.title(f"{title} (Phóng đại: {scale}x)")
    plt.axis('equal')
    plt.show()

def plot_contour(nodes, elements, values, elem_type, title="Contour kết quả", cmap='jet'):
    """
    Xuất đồ thị Contour (Bản đồ màu) cho Chuyển vị hoặc Ứng suất.
    - values: mảng 1D chứa giá trị tại từng NÚT (ví dụ: U_x, U_r, hoặc Von Mises đã nội suy về nút)
    """
    triang = _get_triangulation(nodes, elements, elem_type)
    
    plt.figure(figsize=(8, 6))
    # levels=20 nghĩa là chia thành 20 dải màu
    contour = plt.tricontourf(triang, values, levels=20, cmap=cmap)
    plt.colorbar(contour, label="Value")
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_comparison(r_exact, val_exact, r_fem, val_fem, title="So sánh giải tích và FEM", ylabel="Giá trị"):
    """Xuất đồ thị so sánh kết quả dọc theo bán kính r"""
    plt.figure(figsize=(8, 5))
    
    # Đường giải tích (nét liền đỏ)
    plt.plot(r_exact, val_exact, 'r-', linewidth=2, label='Giải tích (Exact)')
    # Điểm FEM (Chấm xanh)
    plt.plot(r_fem, val_fem, 'bo--', label='Phần tử hữu hạn (FEM)', markersize=5)
    
    plt.title(title)
    plt.xlabel("Bán kính r (mm)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_convergence(mesh_sizes, errors, title="Khảo sát hội tụ (Convergence)"):
    """Xuất đồ thị hội tụ lưới (Số lượng phần tử vs Sai số)"""
    plt.figure(figsize=(8, 5))
    plt.plot(mesh_sizes, errors, 'k-o', linewidth=2, markersize=8)
    
    plt.title(title)
    plt.xlabel("Số lượng phần tử (N)")
    plt.ylabel("Sai số tương đối (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()