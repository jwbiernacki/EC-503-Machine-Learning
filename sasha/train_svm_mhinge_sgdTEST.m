function [W] = train_svm_mhinge_sgdTEST(Xtr, ytr, Delta, T, lambda)
X = Xtr';
[m, d] = size(X);
k = unique(ytr); %returns vector --> help from edward
W = zeros(d,length(k));
for t = 1:T
    random_sample = randi(m,1)';
    x_t = Xtr(:,random_sample);
    y_t = ytr(random_sample);
    learning_rate = 1/(lambda*t);
    y_hat = Delta(y_t,:) + x_t'*W - (x_t'*W(:,y_t));
    [~, arg_max_index] = max(y_hat);
    if arg_max_index == y_t
        W  = W - learning_rate*lambda*W;
    else
        subgradient = zeros(d,length(k));
        subgradient(:,y_t) = -x_t;
        subgradient(:,arg_max_index) = x_t;
        W = W - learning_rate*subgradient - learning_rate * lambda * W;
    end
end
end