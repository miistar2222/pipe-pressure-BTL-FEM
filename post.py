import numpy as np

class PostProcessor:
    def __init__(self, mesh, element, U):
        """
        Khởi tạo Bộ hậu xử lý với dữ liệu đầu vào.
        Lưu trữ các dữ liệu này dưới dạng 'thuộc tính' (attributes) của class.
        """
        self.mesh = mesh
        self.element = element
        self.U = U

    #Hàm tính toán Isoparametric tại tâm phần tử
    def _compute_B_matrix(self, coords):
        if len(coords) == 4: # phần tử Q4 (isoparametric)
            dN_dxi = np.array([[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]])
            J = dN_dxi.T @ coords
            dN_dx = dN_dxi @ np.linalg.inv(J)
            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i] = dN_dx[i,0]; B[1,2*i+1] = dN_dx[i,1]
                B[2,2*i] = dN_dx[i,1]; B[2,2*i+1] = dN_dx[i,0]
            return B
        else:   # phần tử tam giác T3 (isoparametric)
            x1,y1 = coords[0]; x2,y2 = coords[1]; x3,y3 = coords[2]
            A = 0.5 * np.linalg.det([[1,x1,y1], [1,x2,y2], [1,x3,y3]])
            b = [y2-y3, y3-y1, y1-y2]; c = [x3-x2, x1-x3, x2-x1]
            B = (1/(2*A)) * np.array([[b[0],0,b[1],0,b[2],0],
                                      [0,c[0],0,c[1],0,c[2]],
                                      [c[0],b[0],c[1],b[1],c[2],b[2]]])
            return B

    def get_displacements(self):
        #Tính chuyển vị theo hệ Descartes (x, y) và Cực (r, theta)
        #Sử dụng self.U thay vì U truyền từ ngoài vào
        #[u_{x0}, u_{y0}, u_{x1}, u_{y1}, u_{x2}, u_{y2}, ...]
        ux = self.U[0::2]
        uy = self.U[1::2] 
        
        #Chuyển sang hệ cực
        ur = np.zeros_like(ux)  #Chuyển vị hướng tâm
        utheta = np.zeros_like(uy) #Chuyển vị hướng vòng quanh
        
        #Tính theta và chuyển đổi từ (ux, uy) sang (ur, utheta)
        for i, (x, y) in enumerate(self.mesh.nodes):
            theta = np.arctan2(y, x)  
            c, s = np.cos(theta), np.sin(theta) 
            ur[i] = ux[i]*c + uy[i]*s   
            utheta[i] = -ux[i]*s + uy[i]*c  
            
        return ux, uy, ur, utheta

    def get_element_stresses(self):
        #Tính ứng suất hệ Descartes, Cực và Von Mises tại tâm phần tử
        sx_list, sy_list, txy_list = [], [], []    #Ứng suất theo x,y và ứng suất cắt (tau xy)
        vm_list = []    #Von Mises
        sr_list, st_list = [], [] #Ứng suất hướng tâm và vòng quanh
        r_list, theta_list = [], []    #Bán kính và góc của tâm phần tử

        for e in self.mesh.elements:    #lấy 3/4 nút trong 1 phần tử để nhét vô e
            coords = self.mesh.nodes[e] 
            dof = []    
            for n in e: dof += [2*n, 2*n+1] 
            u = self.U[dof]

            B = self._compute_B_matrix(coords)
            
            # Ứng suất XY; với {épilon}=B*{u} và {sigma}=D*{epsilon} --> {sigma} = D * B * {u}
            sx, sy, txy = self.element.mat.D @ (B @ u) #{sigma} = D * B * u, nó sẽ xuất ra sigma_x, sigma_y, tau_xy
            vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2) #Von Mises hệ 2d học từ chvrbd
            
            sx_list.append(sx); sy_list.append(sy); txy_list.append(txy); vm_list.append(vm) 

            # Ứng suất R-Theta
            xc, yc = np.mean(coords, axis=0)    #Tính tọa độ tâm phần tử bằng cách lấy trung bình của các nút
            r = np.hypot(xc, yc) #Bán kính từ gốc tọa độ đến tâm phần tử
            theta = np.arctan2(yc, xc) #Góc
            
            #tính ứng suất hướng tâm và vòng quanh bằng cách chuyển đổi từ hệ Descartes sang hệ cực
            c, s = np.cos(theta), np.sin(theta) #Descartes sang hệ cực
            sr = sx*c**2 + sy*s**2 + 2*txy*s*c  #Ứng suất hướng tâm
            st = sx*s**2 + sy*c**2 - 2*txy*s*c  #Ứng suất hướng vòng
            
            sr_list.append(sr); st_list.append(st)
            r_list.append(r); theta_list.append(theta)

        cartesian_stress = (np.array(sx_list), np.array(sy_list), np.array(txy_list), np.array(vm_list))
        polar_stress = (np.array(r_list), np.array(theta_list), np.array(sr_list), np.array(st_list))
        
        return cartesian_stress, polar_stress
