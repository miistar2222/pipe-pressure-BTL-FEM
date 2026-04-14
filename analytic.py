def lame(r, Ri, Ro, pi, po):
    A = (pi*Ri**2 - po*Ro**2)/(Ro**2-Ri**2)
    B = (Ri**2*Ro**2*(pi-po))/(Ro**2-Ri**2)
    return A-B/r**2, A+B/r**2