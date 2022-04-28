%% Header
% Parker Dunn
% EC 503 Project
% Created 26 April 2022

% Goal: Use "quadprog" to train and test a single optimization classifier
% for multiclass SVM via the dual formulation.

%% Function
function [err, alpha] = train_test_CS_IV_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, nClasses, C)


%% setup
sz = size(Ktrain, 1); % # of training samples
sz_by_k = sz * nClasses;

%% creating f
e = ones(sz_by_k, 1);
for i = 0:(sz-1)
    %for c = Ytrain % e is meant to be a vector with "k" values for each sample
                        % among the "k" values the class that matches the true
                        % class of the sample will be 0
    c = Ytrain(i+1);
    e(i * nClasses + c) = 0;
end

%% creating H
H = repmat(Ktrain, nClasses, nClasses); % This is the Kronecker product form the paper

%% Preparing constraints
% ----- CONSTRAINT ON EACH ALPHA ---------
% (1) when class matches sample: a <= C
% (2) when class does not match sample: a <= 0
% Had to make A and b negative because quadprog uses Ax >= b not <=
A = sparse(eye(sz_by_k));  % OLD: repelem(-eye(sz), 1, nClasses);
b = sparse(C .* [e == 0]);  % <--- "b" is actually similar in structure to e

% ------ CONSTRAINT ON ALPHAS OF EACH CLASS -----
Aeq = sparse(repelem(eye(sz), 1, nClasses));
beq = sparse(zeros(sz, 1));

%% Quadprog and calculating alpha matrix

alpha_vec = quadprog(H, e, A, b, Aeq, beq);

alpha_mat = reshape(alpha_vec, nClasses, sz)';
%After transposing, alpha_mat will have the alphas for each sample in the
%same row

%% Predicting now

% "preds = W * Xtest"
results = Ktest * alpha_mat; % <-- should be sz2 by k, where sz2 is size of testing set

%[~, predictions] = find(results == max(results, [], 2));  % <-- should return just the columns
[m2, d2] = size(results);
predictions = zeros(m2, 1);
for j = 1:m2
   predictions(j) = find(results(j, :) == max(results(j, :))); 
end

%% Returning values

alpha = alpha_mat;
err = mean(predictions ~= Ytest);

end % END FUNCTION