import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from post import PostProcessor

def plot_contour(mesh, ax, values, title):
    polys = [mesh.nodes[e] for e in mesh.elements]
    pc = PolyCollection(polys, array=values, cmap='jet', edgecolor='none')
    ax.add_collection(pc)
    ax.autoscale()
    plt.colorbar(pc, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')

def plot_all(mesh, U, element, scale=200, title="FEM Results"):
    post = PostProcessor(mesh, element, U)
    ux, uy, _, _ = post.get_displacements()
    cartesian_stress, polar_stress = post.get_element_stresses()
    sx, sy, txy, vm = cartesian_stress
    r, theta, sr, st = polar_stress
    
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    axs = axs.flatten()

    for e in mesh.elements:
        pts = mesh.nodes[e + [e[0]]]
        axs[0].plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)
    axs[0].set_title("Original Mesh"); axs[0].set_aspect('equal')

    deformed = mesh.nodes.copy()
    deformed[:, 0] += scale * ux
    deformed[:, 1] += scale * uy
    for e in mesh.elements:
        pts = deformed[e + [e[0]]]
        axs[1].plot(pts[:,0], pts[:,1], 'r-', linewidth=0.5)
    axs[1].set_title(f"Deformation (Scale {scale}x)"); axs[1].set_aspect('equal')

    plot_contour(mesh, axs[2], sx, r"$\sigma_x$ (Cartesian)")
    plot_contour(mesh, axs[3], sy, r"$\sigma_y$ (Cartesian)")
    plot_contour(mesh, axs[4], txy, r"$\tau_{xy}$ (Shear)")
    plot_contour(mesh, axs[5], vm, r"Von Mises Stress")
    plot_contour(mesh, axs[6], sr, r"$\sigma_r$ (Radial Stress)")
    plot_contour(mesh, axs[7], st, r"$\sigma_{\theta}$ (Hoop Stress)")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()