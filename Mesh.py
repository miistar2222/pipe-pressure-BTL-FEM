import numpy as np

class mesh:
    def __init__(self, Ri, Ro, nr, nt, element_type):
        self.nodes = []
        self.elements = []
        self.element_type = element_type    #t3 hay q4

        #vẽ hình tròn
        r_vals = np.linspace(Ri, Ro, nr)    #số vòng tròn (nr)
        theta_vals = np.linspace(0, np.pi/2, nt)    #quay từ 0 tới 90 độ (mô hình 1/4)

        for r in r_vals:    #vẽ
            for t in theta_vals:
                self.nodes.append([r*np.cos(t), r*np.sin(t)])
        self.nodes = np.array(self.nodes)
        
        nt_eff = nt     #số nút trên mỗi vòng tròn
        for i in range(nr-1):           #i hàng
            for j in range(nt-1):       #j cột
                n1 = i*nt_eff + j       #gốc
                n2 = i*nt_eff + j+1     #trên
                n3 = (i+1)*nt_eff + j+1 #chéo trái
                n4 = (i+1)*nt_eff + j   #dưới chéo trái

                if element_type == "Q4":
                    self.elements.append([n1,n2,n3,n4])
                elif element_type == "T3":
                    self.elements.append([n1,n2,n3])
                    self.elements.append([n1,n3,n4])

        self.ndof = len(self.nodes)*2