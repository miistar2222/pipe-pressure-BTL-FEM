% =========================================================
% CHƯƠNG TRÌNH FEM: ỐNG TRỤ CHỊU ÁP SUẤT (PLANE STRAIN)
% =========================================================
clear; clc; close all;

%% 1. THÔNG SỐ ĐẦU VÀO
Ri = 10.0;          % Bán kính trong
Ro = 20.0;          % Bán kính ngoài
p_i = 100.0;        % Áp suất trong
p_o = 0.0;          % Áp suất ngoài
E = 2.0 * 10^5;     % Mô đun đàn hồi
nu = 0.3;           % Hệ số Poisson
elemType = 'Q4';    % Chọn 'T3' hoặc 'Q4'

%% 2. TIỀN XỬ LÝ (PRE-PROCESSING)
% [Thành viên 4 cần viết hàm CreateMesh_Cylinder]
% Giả định ta đã có hàm sinh lưới trả về Nodes và Elements
% [Nodes, Elements] = CreateMesh_Cylinder(Ri, Ro, elemType, 20, 40);
disp('Đang nạp dữ liệu lưới...');

% --- (Phần này bạn chèn dữ liệu Nodes và Elements thực tế vào) ---
% Tạm thời gán dummy để file Main không báo lỗi thiếu biến
Nodes = [10 0; 20 0; 20 10; 10 10]; % Dummy 4 nút
Elements = [1 2 3 4];               % Dummy 1 phần tử Q4
numNodes = size(Nodes, 1);
% -----------------------------------------------------------------

% Khởi tạo vector tải tổng thể F
F = zeros(2*numNodes, 1);
% [Thành viên 4 viết code cộng áp suất pi, po vào các nút biên của F]

%% 3. LẮP RÁP & GIẢI HỆ (SOLVER)
disp('Đang lắp ráp ma trận K tổng thể...');
K = AssembleGlobalK(Nodes, Elements, E, nu, elemType);

disp('Đang áp dụng điều kiện biên...');
% [Thành viên 4 xác định các bậc tự do bị khóa (fixed_dofs)]
fixed_dofs = [2, 4]; % Dummy ngàm
free_dofs = setdiff(1:(2*numNodes), fixed_dofs);

disp('Đang giải hệ phương trình...');
U = zeros(2*numNodes, 1);
% Giải K * U = F tại các bậc tự do tự do
U(free_dofs) = K(free_dofs, free_dofs) \ F(free_dofs);

%% 4. HẬU XỬ LÝ & ĐỒ HỌA (POST-PROCESSING)
disp('Đang tính toán ứng suất và giải tích...');
% [Thành viên 1 & 6 viết hàm StressRecovery để tính ứng suất]

% Kiểm chứng với lời giải Lame (Thành viên 5)
r_test = linspace(Ri, Ro, 50);
[sig_r_ana, sig_th_ana, ur_ana] = AnalyticalSolution(r_test, Ri, Ro, p_i, p_o, E, nu);

% Vẽ đồ thị (Thành viên 6)
figure;
plot(r_test, sig_r_ana, 'r-', 'LineWidth', 2.0); hold on;
% Lấy giá trị FEM thu được để plot chồng lên (dùng hàm plot dạng điểm 'bo')
xlabel('Bán kính r'); ylabel('Ứng suất hướng tâm \sigma_r');
title('So sánh Giải tích và FEM');
grid on; legend('Giải tích Lame', 'FEM');

disp('Hoàn tất chương trình!');