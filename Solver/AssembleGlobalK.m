function K = AssembleGlobalK(Nodes, Elements, E, nu, elemType)
    numNodes = size(Nodes, 1);
    numElems = size(Elements, 1);
    t = 1.0;
    
    % Khởi tạo các mảng cho ma trận thưa (Sparse)
    I = []; J = []; V = [];
    
    for e = 1:numElems
        elemNodes = Elements(e, :);
        nodeCoords = Nodes(elemNodes, :);
        
        if strcmp(elemType, 'T3')
            ke = StiffnessT3(nodeCoords, E, nu, t);
            dofs = 6;
        elseif strcmp(elemType, 'Q4')
            ke = StiffnessQ4(nodeCoords, E, nu, t);
            dofs = 8;
        end
        
        % Ánh xạ bậc tự do (Degrees of Freedom)
        sctr = zeros(1, dofs);
        for k = 1:length(elemNodes)
            sctr(2*k-1) = 2*elemNodes(k) - 1; % Phương x
            sctr(2*k)   = 2*elemNodes(k);     % Phương y
        end
        
        % Nạp dữ liệu vào I, J, V để tạo Sparse
        [idx, jdx] = meshgrid(sctr, sctr);
        I = [I; idx(:)];
        J = [J; jdx(:)];
        V = [V; ke(:)];
    end
    
    K = sparse(I, J, V, 2*numNodes, 2*numNodes);
end