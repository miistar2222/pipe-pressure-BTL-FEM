import numpy as np

class Q4:        #hàm dạng cho phần tử tứ giác
    def __init__(self, mat):
        self.mat = mat

    def stiffness(self, coords):    #coords: tọa độ x, y của 4 nút
        Ke = np.zeros((8,8))        #trả về ma trận 8x8
        D = self.mat.D              #lấy E và nu từ Material
        gauss = [-1/np.sqrt(3), 1/np.sqrt(3)]   #gauss bậc 2, tứ giác 4 điểm nên dùng gauss bậc 2

        for xi in gauss:
            for eta in gauss:               #đạo hàm N_i theo e và n
                dN_dxi = np.array([         #tọa độ của 4 nút
                    [-(1-eta), -(1-xi)],    #e=-1; n=-1
                    [ (1-eta), -(1+xi)],    #e= 1; n=-1
                    [ (1+eta),  (1+xi)],    #e= 1; n= 1
                    [-(1+eta),  (1-xi)]     #e= 1; n= 1
                ]) / 4                      

                J = dN_dxi.T @ coords       #ma trận jacobian (2x2): lấy chuyển vị (2x4) x tọa độ coords (4x2)
                dN_dx = dN_dxi @ np.linalg.inv(J)   #tính biến dạng bằng đạo hàm

                B = np.zeros((3,8)) #ma trận biến dạng - chuyển vị   sắp đạo hàm về đúng vị tri để nhân với vct chuyển vị
                for i in range(4):  #0: biến dạng dọc trục X
                                    #1: biến dạng dọc trục Y
                                    #2: biến dạng cắt (tauxy)
                    B[0,2*i]   = dN_dx[i,0]     #eps_x
                    B[1,2*i+1] = dN_dx[i,1]     #eps_y
                    B[2,2*i]   = dN_dx[i,1]     #gamma_xy
                    B[2,2*i+1] = dN_dx[i,0]     #gamma_xy

                Ke += 1 * B.T @ D @ B * np.linalg.det(J)
                #[K^e]=W  [B]^T [D]  [B]  det(j)
        return Ke

class T3:
    def __init__(self, mat):
        self.mat = mat

    def stiffness(self, coords):
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]

        #diện tích A của phần tử tam giác
        A = 0.5*np.linalg.det([[1,x1,y1],[1,x2,y2],[1,x3,y3]]) 

        b = [y2-y3, y3-y1, y1-y2]
        c = [x3-x2, x1-x3, x2-x1]

        B = (1/(2*A))*np.array([
            [b[0],  0,      b[1],   0,      b[2],   0   ],
            [0,     c[0],   0,      c[1],   0,      c[2]],
            [c[0],  b[0],   c[1],   b[1],   c[2],   b[2]]
        ])

        return B.T @ self.mat.D @ B * A * 2
