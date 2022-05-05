%% Header
% Parker Dunn
% EC 503 Project

% Goal: This function performs k-fold cross validation for an single optimization SVM in
% order to identify the optimal parameters.

% Inputs: 
%   (1) X (samples)
%   (2) Y (classes)
%   (3) # of folds
%   (4) type of kernel to use
%   (5) classes   <-- the number of classes
%   () lambda/soft-margin parameter
%   () kernel specific parameter

% Outputs: Estimated error from the k-fold validation process

%% Function start
function err = k_fold_cross_validation_SVM(X, Y, folds, kernel, nClasses, C, param)

[m, ~] = size(X);
errors = zeros(1, folds);

% Fold iterations
for k = 1:folds
    %idx_train = (mod(1:m,folds) == (k-1));
        % mod(1:m, folds) -> splits samples into "folds" groups
        % Overall: The statement uses logical indexing to select one of the
        % groups
    
    idx_train = (mod(1:m, 3) ~= k); % <-- should always select 2/3 of the training set
    
    Xtrain = X(idx_train, :);
    Ytrain = Y(idx_train);
    Xtest = X(~idx_train, :);
    Ytest = Y(~idx_train);
    
    % Generating the kernel matrices
    Ktrain = compute_kernel_matrix(Xtrain, Xtrain, kernel, param);
    Ktest = compute_kernel_matrix(Xtest, Xtrain, kernel, param);
    
    % FINALLY, train and test of SVM classifier
    %[fold_err, alpha_matrix] = train_test_ADO_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, nClasses, C);
    [fold_err, alpha_mat] = train_test_CS_IV_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, nClasses, C);
    errors(k) = fold_err;
end

% Estimating error current parameters
% err = mean(errors);
err = mean(errors, 'omitnan');

end