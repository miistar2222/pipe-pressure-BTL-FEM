import numpy as np

class ChuyenViXY:
    def __init__(self, U_full, nodes):
        self.U = U_full
        self.num_nodes = len(nodes)
        
    def get_displacements(self):
        """
        Trích xuất chuyển vị theo phương X và Y cho từng nút
        Return: ux, uy (các mảng numpy 1D)
        """
        ux = self.U[0::2]  # Lấy các giá trị ở vị trí chẵn (0, 2, 4...)
        uy = self.U[1::2]  # Lấy các giá trị ở vị trí lẻ (1, 3, 5...)
        return ux, uy