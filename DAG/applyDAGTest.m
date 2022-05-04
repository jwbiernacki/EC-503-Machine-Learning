clear;
clc;

%%
%Clean data and compute kernel
kernelSelect = 1; %select 1 for linear, 2 for poly, 3 for Gaussian
cleanDataTest;

%%
%Train

first = 7;
second = 7;

[newC, tTrainEnd] = justTrainTest(xTrMain, yTrMain, first, second);

%%
%Validate

[resultsVal, tValEnd] = justValTest(Xval, first, second, newC);

%%
%Compute Error

[validationAccuracy, bestC, idx] = computeErrorTest(resultsVal, Yval);

%%
%Test

[resultsTest, tTestEnd] = runTestTest(Xtest, newC{7});

%%
%Compute Error

testAccuracy = computeErrorV2(resultsTest, Ytest);

%%
fprintf('Best C: %.3f\n', bestC);
fprintf('Training time: %.5f\n', tTrainEnd);
fprintf('Validation time: %.5f\n', tTrainEnd);
fprintf('Validation accuracy: %.3f\n', validationAccuracy);
fprintf('Testing time: %.5f\n', tTestEnd);
fprintf('Testing accuracy: %.3f\n', testAccuracy);