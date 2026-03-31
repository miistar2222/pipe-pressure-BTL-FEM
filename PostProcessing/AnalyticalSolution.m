function [sigma_r, sigma_theta, u_r] = AnalyticalSolution(r, Ri, Ro, pi, po, E, nu)
    % r có thể là một mảng các giá trị bán kính
    
    % Hệ số Lame
    A = (pi * Ri^2.0 - po * Ro^2.0) / (Ro^2.0 - Ri^2.0);
    B = (pi - po) * (Ri^2.0 * Ro^2.0) / (Ro^2.0 - Ri^2.0);
    
    % Ứng suất hướng tâm và tiếp tuyến
    sigma_r = A - B ./ (r.^2.0);
    sigma_theta = A + B ./ (r.^2.0);
    
    % Chuyển vị hướng tâm (Plane Strain)
    % Công thức: u_r = (1+nu)/E * [ (1-2*nu)*A*r + B/r ]
    u_r = ((1.0 + nu) / E) .* ((1.0 - 2.0 * nu) .* A .* r + B ./ r);
end