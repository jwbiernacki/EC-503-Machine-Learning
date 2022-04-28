function [W, B, LUT] = train_svm_one_vs_one(k, Xtr, ytr, C)
%
%
%
%
    Xtr         = [ones(1, size(Xtr, 2)); Xtr];
    [d, m]      = size(Xtr);
    W           = zeros(d, k * (k - 1) / 2);
    B           = zeros(1, k * (k - 1) / 2);
    LUT         = zeros(2, k * (k - 1) / 2);
    idx_counter = 1;

    for i = 1 : 1 : k
        for j = i + 1 : 1 :  k
            fprintf('Training %d and %d \n %d out of %d \n', i, j, idx_counter, k * (k - 1) / 2)
  
            % Look Up Table: Which two classes are being trained on
            LUT(1, idx_counter) = i;
            LUT(2, idx_counter) = j;

            % Extract samples only from class i and j from Xtr
            i_idx = find(ytr == i)';
            j_idx = find(ytr == j)';
            y_tr = ytr( sort([i_idx j_idx]) );
            X_tr = Xtr(:, sort([i_idx, j_idx]))';
            [m, d] = size(X_tr);
            
            % Binarize
            y_tr(y_tr == i) = 1;
            y_tr(y_tr == j) = -1;

            % Construct quadprog parameters
            H = (X_tr * X_tr') .* (y_tr * y_tr');
            f = -ones(1,m);
            
            A=[];
            b2=[];
            
            LB = zeros(1, m);
            UB = ones(1, m) * C;
            
            % Quadprog
            [alpha]=quadprog(H,f,A,b2,[],[],LB,UB);

            % Find W
%             w = zeros(d,1);
%             for l = 1 : m
%                 w = w + alpha(l) * y_tr(l) * X_tr(l,:)';
%                 b = b + 1 - X_tr(l, :) * w;
%             end
            w = X_tr'*(alpha.*y_tr);
            W(:, idx_counter) = w;
            idx_counter = idx_counter + 1;
        end
    end
    B = W(1, :);
    W(1, :) = [];
end