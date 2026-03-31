function ke = StiffnessT3(nodes, E, nu, t)
    % nodes: ma trận 3x2 chứa tọa độ [x1 y1; x2 y2; x3 y3]
    % E: Mô đun đàn hồi, nu: Hệ số Poisson, t: Chiều dày (t = 1.0)
    
    % Ma trận vật liệu D cho bài toán Biến dạng phẳng (Plane Strain)
    c = E / ((1 + nu) * (1 - 2.0 * nu));
    D = c * [1 - nu, nu,     0;
             nu,     1 - nu, 0;
             0,      0,      (1 - 2.0 * nu) / 2.0];
         
    % Tọa độ các nút
    x1 = nodes(1,1); y1 = nodes(1,2);
    x2 = nodes(2,1); y2 = nodes(2,2);
    x3 = nodes(3,1); y3 = nodes(3,2);
    
    % Tính diện tích phần tử A
    J = [1 1 1; x1 x2 x3; y1 y2 y3];
    A = 0.5 * det(J);
    
    % Các hệ số b và c
    b1 = y2 - y3; c1 = x3 - x2;
    b2 = y3 - y1; c2 = x1 - x3;
    b3 = y1 - y2; c3 = x2 - x1;
    
    % Ma trận liên hệ biến dạng - chuyển vị B
    B = (1 / (2.0 * A)) * [b1  0  b2  0  b3  0;
                           0  c1  0  c2  0  c3;
                           c1 b1  c2 b2  c3 b3];
                       
    % Ma trận độ cứng phần tử
    ke = B' * D * B * t * A;
end