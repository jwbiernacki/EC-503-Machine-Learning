%% Header
% Parker Dunn
% EC 503 - HW5
% Problem 5.1 - Part (c)

function [w] = train_svm_mhinge_sgd(Xtr, ytr, Delta, T, lambda)
%% Description of Function
% Xtr -> input matrix
% ytr -> vector of labels
% Delta -> matrix of the costs
% Lambda -> regularization weight

% Steps:
% (1) initialize Wmat
% (2) Start loop from 1 to T
    % a. get prediction for sample ... multiply it by Wmat i think???
    % b. update Wmat using subgradient (using formula provided in class)


% k -> # of classes
% d -> # of parameters/features

%% Start of function actions

% NOTE: Currently assuming that Xtr has samples as columns (as indicated by
% the assignment document)

[d, m] = size(Xtr);
% ytr should be 1 x m

k = length(unique(ytr));

Wmat = zeros(d, k);
% k columns ... one for each class
% d rows ... # of features

for t = 1:T
    
    % Select a random sample from Xtr
    i = randi([1, m]);
    
    % Select column/sample & corresponding label
    x_i = Xtr(:,i);
    y_i = ytr(i);

    eta = 1/(t*lambda);
    
    % UPDATING Wmat
    
    % (a) Finding y_hat
    trueCol = Delta(:,y_i); % trueCol is k x 1  <---- Delta is k x k
    % Assuming ... cols = True label AND rows = predicted label
    W_diff = Wmat - Wmat(:,y_i);  % W_diff is d x k; all w_y' - w_yi calcs at once
    loss = trueCol + (W_diff' * x_i);
    
    % finding and selecting the max y'
    y_hat = 1;
    for i = 1:k
        if (loss(i) > loss(y_hat))
            y_hat = i;
        end
    end

    % (b) Calculate subgradient and update Wmat
    G_t = zeros(d, k);
    G_t(:,y_hat) = G_t(:, y_hat) + x_i;
    G_t(:,y_i) = G_t(:, y_i) - x_i;

    Wmat = (1-(1/t)).*Wmat - eta.*G_t;
    
    %fprintf("SGD Iteration %d. Here is W:\n",t)
    %disp(Wmat)
    
end

w = Wmat;

end