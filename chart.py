import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def _plot_contour(self, ax, values, title):
    #Hàm vẽ màu cho phần tử dựa trên giá trị ứng suất hoặc chuyển vị
    polys = [self.mesh.nodes[e] for e in self.mesh.elements]
    pc = PolyCollection(polys, array=values, cmap='jet', edgecolor='none')
    ax.add_collection(pc)
    ax.autoscale()
    plt.colorbar(pc, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')

def plot_all(self, scale=200, title="FEM Results"):
    #Hàm chính để xuất toàn bộ đồ thị bao gồm lưới ban đầu, lưới biến dạng và các biểu đồ ứng suất
    ux, uy, _, _ = self.get_displacements()
    cartesian_stress, polar_stress = self.get_element_stresses()
    sx, sy, txy, vm = cartesian_stress
    r, theta, sr, st = polar_stress
    
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    axs = axs.flatten()

    for e in self.mesh.elements:
        pts = self.mesh.nodes[e + [e[0]]]
        axs[0].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    axs[0].set_title("Original Mesh"); axs[0].set_aspect('equal')

    deformed = self.mesh.nodes.copy()
    deformed[:, 0] += scale * ux
    deformed[:, 1] += scale * uy
    for e in self.mesh.elements:
        pts = deformed[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'r-', linewidth=0.5)
    axs[1].set_title(f"Deformation (Scale {scale}x)"); axs[1].set_aspect('equal')

    # Gọi hàm vẽ màu của class
    self._plot_contour(axs[2], sx, r"$\sigma_x$ (Cartesian)")
    self._plot_contour(axs[3], sy, r"$\sigma_y$ (Cartesian)")
    self._plot_contour(axs[4], txy, r"$\tau_{xy}$ (Shear)")
    self._plot_contour(axs[5], vm, r"Von Mises Stress")
    self._plot_contour(axs[6], sr, r"$\sigma_r$ (Radial Stress)")
    self._plot_contour(axs[7], st, r"$\sigma_{\theta}$ (Hoop Stress)")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()