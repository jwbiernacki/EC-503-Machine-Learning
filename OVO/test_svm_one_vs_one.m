clear
clc
close all
f1 = @()test_svm();
Trainvaltime = timeit(f1)

function test_svm()
    
    
    % Load data and Initialize Parameters
    load('satimage_data.mat')
    
    rand = randperm(numel(ytr));
    med = median(Xtr,'omitnan');
    Xtr = fillmissing(Xtr,'constant',med);
    Xtr = Xtr(rand, :);
    ytr = ytr(rand);
%     Xval = Xtr(1 : 1326, :);
%     Xtr(1 : 1326, :) = [];
%     yval = ytr(1 : 1326);
%     ytr(1 : 1326) = [];
%     
%     
    k = numel(unique(ytr));
    vote = zeros(1, k);
    test_err = 0;
    folds = 10;
    cnt = 1;
%     
%     
%     
%     % Accuracy Error 1 - Test_Err
%     acc_err = zeros(1, folds);
%     
%     % Test Lambda
%     for lambda = (1:0.05:2).^2
%         disp(lambda)
%         % Train Dual SVM Step
%         [W, B, LUT] = train_svm_one_vs_one(k, Xtr', ytr, lambda);
%         
%         % Validation Step
%         [m, d] = size(Xval);
%     
%         for j = 1 : m
%             x_j = Xval(j, :);     % Extract Each Val sample 1x4
%             voting = sign((x_j * W) + B); % x_j 15x4 and W is 4x3 = voting 15x3
%             for l = 1 : size(voting, 2) % Loop Through Each k classes
%                 if voting(l) == 1 % If scalar is pos
%                     vote(LUT(1, l)) = vote(LUT(1, l)) + 1;
%                 else
%                     vote(LUT(2, l)) = vote(LUT(2, l)) + 1; % If scalar is neg
%                 end
%             end
%             [~, y_hat] = max(vote);
%             if yval(j) ~= y_hat
%                 test_err = test_err + 1;
%             end
%             vote = vote * 0;
%         end
%        acc_err(cnt) = 1 - ( test_err / numel(yval) );
%        test_err = 0;
%        cnt = cnt + 1;
%     
%     end
%     
%     [val, idx] = max(acc_err)
    
    lambda = 1.1025;
    
    % Validation Step
    [m, d] = size(Xte);
    
    [W, B, LUT] = train_svm_one_vs_one(k, Xtr', ytr, lambda);
    
    for j = 1 : m
        x_j = Xte(j, :);     % Extract Each Val sample 1x4
        voting = sign((x_j * W) + B); % x_j 15x4 and W is 4x3 = voting 15x3
        for l = 1 : size(voting, 2) % Loop Through Each k classes
            if voting(l) == 1 % If scalar is pos
                vote(LUT(1, l)) = vote(LUT(1, l)) + 1;
            else
                vote(LUT(2, l)) = vote(LUT(2, l)) + 1; % If scalar is neg
            end
        end
        [~, y_hat] = max(vote);
        if yte(j) ~= y_hat
            test_err = test_err + 1;
        end
        vote = vote * 0;
    end
    acc_err(cnt) = 1 - ( test_err / numel(yte) );
end