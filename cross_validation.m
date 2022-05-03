function err = cross_validation(X, Y, folds, kernel, param, lambda)

[m]=size(X,1);

errs=zeros(1,folds);

for j=1:folds
    idx_train=(mod(1:m,folds)==j-1);
    Xtrain=X(idx_train,:);
    Xtest=X(~idx_train,:);
    Ytrain=Y(idx_train);
    Ytest=Y(~idx_train);
    
    Ktrain=compute_kernel_matrix(Xtrain,Xtrain,kernel,param);
    Ktest=compute_kernel_matrix(Xtest,Xtrain,kernel,param);
    
    errs(j)=train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambda);
end

err=mean(errs);
end

