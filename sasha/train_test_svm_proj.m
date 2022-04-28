clear
load satimage_data.mat

[m,d]=size(Xtr);

med = median(Xtr,'omitnan');
Xtr = fillmissing(Xtr,'constant',med);
med = median(Xte,'omitnan');
Xte = fillmissing(Xte,'constant',med);


%split up the data
rand_num = randperm(4420);
Xtrain = Xtr(rand_num(1:3094),:)'; 
ytrain = ytr(rand_num(1:3094))';
Xval = Xtr(rand_num(3095:end),:)';
yval = ytr(rand_num(3095:end))';

Delta = ones(6)-eye(6);
T = 50;
lambdas=10.^(-15:10);
accuracy=zeros(1,numel(lambdas));
for i=1:numel(lambdas)
    accuracy(i) = 0;
    W = train_svm_mhinge_sgdPARKER(Xtrain,ytrain,Delta,T,lambdas(i));
    for j=1:size(Xval,2)
    pred_val=Xval(:, j)'*W;
    [~,mx_idx]=max(pred_val);
    ypred(j)=mx_idx;
    end
    accuracy(i) = mean(ypred == yval);
end
[~,idx]=max(accuracy);

% for i = 1:size(Xte,2)
%     pred_test = Xte(:, j)' * W;
%     [~,mx_idx_test] = max(pred_test);
%     ypred_te(i) = mx_idx_test;
% end
% error_te = mean(ypred_te == yte);