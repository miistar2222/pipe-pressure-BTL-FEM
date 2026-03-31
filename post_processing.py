import numpy as np
from elements import get_d_matrix, shape_functions_q4

def calculate_stresses(nodes, elements, U, elem_type, E, nu):
    D = get_d_matrix(E, nu)
    stresses = [] # Lưu [sigma_x, sigma_y, tau_xy, sigma_vM, r, theta]

    for elem in elements:
        node_coords = nodes[elem]
        # Lấy chuyển vị của các nút thuộc phần tử này
        u_e = []
        for node_id in elem:
            u_e.extend([U[2*node_id], U[2*node_id + 1]])
        u_e = np.array(u_e)

        # Tính tại tâm phần tử (xi=0, eta=0 cho Q4 hoặc trọng tâm cho T3)
        if elem_type == 'T3':
            # Ma trận B của T3 là hằng số (lấy lại từ elements.py)
            x1, y1 = node_coords[0]; x2, y2 = node_coords[1]; x3, y3 = node_coords[2]
            two_area = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            a = [y2 - y3, y3 - y1, y1 - y2]
            b = [x3 - x2, x1 - x3, x2 - x1]
            B = np.array([[a[0], 0, a[1], 0, a[2], 0],
                          [0, b[0], 0, b[1], 0, b[2]],
                          [b[0], a[0], b[1], a[1], b[2], a[2]]]) / two_area
            center = np.mean(node_coords, axis=0)
        else: # Q4
            N, dN_dxi_eta = shape_functions_q4(0, 0)
            J = dN_dxi_eta @ node_coords
            invJ = np.linalg.inv(J)
            dN_dx_dy = invJ @ dN_dxi_eta
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k] = dN_dx_dy[0, k]; B[1, 2*k+1] = dN_dx_dy[1, k]
                B[2, 2*k] = dN_dx_dy[1, k]; B[2, 2*k+1] = dN_dx_dy[0, k]
            center = N @ node_coords

        # Tính ứng suất sigma = D * B * u_e
        sigma = D @ B @ u_e
        sx, sy, txy = sigma[0], sigma[1], sigma[2]
        
        # Tính Von Mises (Biến dạng phẳng)
        sz = nu * (sx + sy)
        vM = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2 + 6*txy**2))
        
        # Chuyển sang tọa độ cực để lấy sigma_r, sigma_theta
        r_val = np.linalg.norm(center)
        theta_val = np.arctan2(center[1], center[0])
        sr = sx*np.cos(theta_val)**2 + sy*np.sin(theta_val)**2 + 2*txy*np.sin(theta_val)*np.cos(theta_val)
        st = sx*np.sin(theta_val)**2 + sy*np.cos(theta_val)**2 - 2*txy*np.sin(theta_val)*np.cos(theta_val)
        
        stresses.append([sx, sy, txy, vM, sr, st, r_val])

    return np.array(stresses)