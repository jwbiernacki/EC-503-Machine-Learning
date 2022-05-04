function err = cross_validation(X, Y, folds, lambda)

[m]=size(X,1);
T = 10000;
load vowel.mat
errs=zeros(1,folds);
    for j=1:folds
        idx_train=(mod(1:m,folds)==j-1);
        Xtrain=Xtr(idx_train,:)';
        Xtest=Xtr(~idx_train,:)';
        Ytrain=ytr(idx_train)';
        Ytest=ytr(~idx_train)';
        numclass = max(Ytrain);
        Delta = ones(numclass)-eye(numclass);
        tic
            W = train_svm_mhinge_sgdPARKER(Xtrain,Ytrain,Delta,T,lambda);
            ypred = zeros(size(X,2),1)';
            for i=1:size(Xtest,2)
                pred_val=Xtest(:, i)'*W;
                [~,mx_idx]=max(pred_val);
                ypred(i)=mx_idx;
            end
            errs(j) = mean(ypred ~= Ytest);
        end

err=mean(errs);
end