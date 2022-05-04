clear
close all
folds=10;
lambdas=10.^(-10:5);
T = 10000;
% Linear
error=zeros(1,numel(lambdas));
load vowel.mat
for i=1:numel(lambdas)
    error(i) = cross_validation(Xtr, ytr, folds, lambdas(i));
end