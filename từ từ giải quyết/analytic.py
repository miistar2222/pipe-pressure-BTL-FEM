import numpy as np

def get_lame_results(r, Ri, Ro, pi, po, E, nu):
    """
    Tính toán đầy đủ lời giải giải tích Lame cho biến dạng phẳng.
    """
    # Các hằng số Lame
    A = (pi * Ri**2 - po * Ro**2) / (Ro**2 - Ri**2)
    B = (Ri**2 * Ro**2 * (pi - po)) / (Ro**2 - Ri**2)
    
    # 1. Ứng suất hệ cực
    sr = A - B / r**2
    st = A + B / r**2
    
    # 2. Chuyển vị hướng tâm (Plane Strain)
    ur = (r * (1 + nu) / E) * (A * (1 - 2 * nu) + B / r**2)
    
    # 3. Ứng suất hệ Descartes (Xét trên trục X: theta = 0)
    # Tại theta = 0: sx = sr, sy = st, txy = 0
    sx = sr
    sy = st
    txy = np.zeros_like(r)
    
    # 4. Ứng suất Von Mises (Tính theo công thức 2D trong post.py để đồng nhất)
    vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    
    return {
        'ur': ur,
        'sr': sr, 'st': st,
        'sx': sx, 'sy': sy,
        'vm': vm
    }