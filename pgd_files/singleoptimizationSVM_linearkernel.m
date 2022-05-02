%% Header
% Parker Dunn
% Ec 503 Project
% 28 April 2022

% Testing "train_test_CS_IV_svm_kernel" with multiple kernels

%% Setup/preparation

clear
close all

fid = fopen("results.txt",'a');

%load satimage_data.mat
%load iris_data.mat   -- skipped for now
%load wine_data.mat   -- skipped for now
load letter_data.mat

%dataset = "satimage";
%dataset = "iris";    -- skipped for now
%dataset = "wine";    -- skipped for now
dataset = "letter";



date = "2 May 2022";
fprintf(fid, "\n----------\nDate: %s\nTesting Linear kernel - Dataset = %s\n", date, dataset);

fprintf(fid, "* NOTE * - Only 1/3 of the data was used due to complexity of this algorithm.\n");

[m, d]=size(Xtr);
[m2, d2]=size(Xte);

% IF ONLY SOME OF THE DATASET IS NEEDED
Xtr = Xtr(mod(1:m,6) == 0, :);
Xte = Xte(mod(1:m2,6) == 0, :);
ytr = ytr(mod(1:m,6) == 0);
yte = yte(mod(1:m2,6) == 0);

med = median(Xtr,'omitnan');
Xtr = fillmissing(Xtr,'constant',med);
med = median(Xte,'omitnan');
Xte = fillmissing(Xte,'constant',med);

folds = 1;
classes = unique(ytr);

vals = -6:-4;

C = 10.^(vals);

%% Validation with Linear Kernel
tic
errors = zeros(numel(C), 1);
for i = 1:numel(C)
    % param = anything for linear kernel
    % "1" argument ==> linear kernel
    errors(i) = k_fold_cross_validation_SVM(Xtr, ytr, folds, 1, numel(classes), C(i), 0);
end
fprintf(fid, "Time to train/validate with linear kernel and THREE C parameters: %.4f\n", toc);


figure(1)
plot(vals, 100*(1-errors), '*k')
xlabel("C")
ylabel("Accuracy")
axis padded

[~,idx]=min(errors);
fprintf(fid, 'Best C by cross-validation with linear kernel: %e\n', C(idx));

% Testing procedure for this kernel
tic
Ktrain=compute_kernel_matrix(Xtr,Xtr,1,0);
Ktest=compute_kernel_matrix(Xte,Xtr,1,0);
[test_err, alpha] = train_test_CS_IV_svm_kernel(Ktrain, Ktest, ytr, yte, numel(classes), C(idx));
fprintf(fid, "Time to test with LINEAR kernel: %.4f\n", toc);
fprintf(fid, "Testing Error with LINEAR kernel: %.3f\n", test_err);