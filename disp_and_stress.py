import numpy as np

def get_displacements(mesh, U):
    #Tính chuyển vị theo hệ Descartes (x, y) và Cực (r, theta)
    ux = U[0::2]
    uy = U[1::2] 
    
    ur = np.zeros_like(ux)
    utheta = np.zeros_like(uy)
    
    for i, (x, y) in enumerate(mesh.nodes):
        theta = np.arctan2(y, x)  
        c, s = np.cos(theta), np.sin(theta) 
        ur[i] = ux[i]*c + uy[i]*s   
        utheta[i] = -ux[i]*s + uy[i]*c  
        
    return ux, uy, ur, utheta


def get_element_stresses(mesh, element, U):
    #Tính ứng suất hệ Descartes, Cực và Von Mises tại tâm phần tử
    # Các list lưu trữ kết quả
    sx_list, sy_list, txy_list, vm_list = [], [], [], []
    sr_list, st_list, r_list, theta_list = [], [], [], []

    for e in mesh.elements:
        coords = mesh.nodes[e] 
        dof = []    
        for n in e: 
            dof += [2*n, 2*n+1] 
        u_elem = U[dof]

        # --- Tính ma trận B (Logic từ hàm _compute_B_matrix cũ) ---
        if len(coords) == 4: # Q4
            dN_dxi = np.array([[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]])
            J = dN_dxi.T @ coords
            dN_dx = dN_dxi @ np.linalg.inv(J)
            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i] = dN_dx[i,0]; B[1,2*i+1] = dN_dx[i,1]
                B[2,2*i] = dN_dx[i,1]; B[2,2*i+1] = dN_dx[i,0]
        else: # T3
            x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
            A = 0.5 * np.linalg.det([[1,x1,y1], [1,x2,y2], [1,x3,y3]])
            b = [y2-y3, y3-y1, y1-y2]
            c_val = [x3-x2, x1-x3, x2-x1]
            B = (1/(2*A)) * np.array([[b[0],0,b[1],0,b[2],0],
                                      [0,c_val[0],0,c_val[1],0,c_val[2]],
                                      [c_val[0],b[0],c_val[1],b[1],c_val[2],b[2]]])
        
        # --- Tính toán Ứng suất ---
        # {sigma} = D * B * {u}
        sx, sy, txy = element.mat.D @ (B @ u_elem)
        vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
        
        sx_list.append(sx); sy_list.append(sy); txy_list.append(txy); vm_list.append(vm) 

        # Tọa độ tâm phần tử
        xc, yc = np.mean(coords, axis=0)
        r = np.hypot(xc, yc)
        theta = np.arctan2(yc, xc)
        
        # Chuyển sang hệ cực
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        sr = sx*cos_t**2 + sy*sin_t**2 + 2*txy*sin_t*cos_t
        st = sx*sin_t**2 + sy*cos_t**2 - 2*txy*sin_t*cos_t
        
        sr_list.append(sr); st_list.append(st)
        r_list.append(r); theta_list.append(theta)

    cartesian_stress = (np.array(sx_list), np.array(sy_list), np.array(txy_list), np.array(vm_list))
    polar_stress = (np.array(r_list), np.array(theta_list), np.array(sr_list), np.array(st_list))
    
    return cartesian_stress, polar_stress