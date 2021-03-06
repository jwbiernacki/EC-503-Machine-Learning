clear
load shuttle.mat

[m,d]=size(Xtr);

% split up the data
shufnum = floor(.7*m);
rand_num = randperm(m);
Xtrain = Xtr(rand_num(1:shufnum),:)'; 
ytrain = ytr(rand_num(1:shufnum))';
Xval = Xtr(rand_num(shufnum+1:end),:)';
yval = ytr(rand_num(shufnum+1:end))';
Xte = Xte';
yte = yte';

numclass = max(ytr);
Delta = ones(numclass)-eye(numclass);
T = 1e6;
lambdas=10.^(-15:10);
accuracy=zeros(1,numel(lambdas));
tic
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
W = train_svm_mhinge_sgdPARKER(Xtrain,ytrain,Delta,T,lambdas(idx));
fprintf("Training and validation time is ")
toc
tic
for i = 1:size(Xte,2)
    pred_test = Xte(:, i)' * W;
    [~,mx_idx_test] = max(pred_test);
    ypred_te(i) = mx_idx_test;
end
accuracy_te = mean(ypred_te == yte);
fprintf("Testing time is ")
toc

mat=zeros(numclass,numclass);
for i=1:numel(yte)
    mat(yte(i),ypred_te(i))=mat(yte(i),ypred_te(i))+1;
end