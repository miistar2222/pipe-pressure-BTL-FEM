function ke = StiffnessQ4(nodes, E, nu, t)
    % nodes: ma trận 4x2 chứa tọa độ 4 nút
    
    % Ma trận D (Plane Strain)
    c = E / ((1 + nu) * (1 - 2.0 * nu));
    D = c * [1 - nu, nu,     0;
             nu,     1 - nu, 0;
             0,      0,      (1 - 2.0 * nu) / 2.0];
         
    % Tọa độ điểm Gauss (2x2) và trọng số
    gauss_pts = [-1/sqrt(3.0), 1/sqrt(3.0)];
    weights = [1.0, 1.0];
    
    ke = zeros(8, 8);
    
    % Vòng lặp tích phân Gauss
    for i = 1:2
        for j = 1:2
            xi = gauss_pts(i);
            eta = gauss_pts(j);
            w = weights(i) * weights(j);
            
            % Đạo hàm hàm dạng theo xi và eta
            dN_dxi = 0.25 * [-(1-eta), 1-eta, 1+eta, -(1+eta)];
            dN_deta = 0.25 * [-(1-xi), -(1+xi), 1+xi, 1-xi];
            dN = [dN_dxi; dN_deta];
            
            % Ma trận Jacobi
            J = dN * nodes;
            detJ = det(J);
            
            % Đạo hàm hàm dạng theo x và y
            dN_dxy = J \ dN;
            
            % Thiết lập ma trận B
            B = zeros(3, 8);
            for k = 1:4
                B(1, 2*k-1) = dN_dxy(1, k);
                B(2, 2*k)   = dN_dxy(2, k);
                B(3, 2*k-1) = dN_dxy(2, k);
                B(3, 2*k)   = dN_dxy(1, k);
            end
            
            % Cộng dồn vào ke
            ke = ke + B' * D * B * t * detJ * w;
        end
    end
end