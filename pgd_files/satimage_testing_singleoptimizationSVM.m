%% Header
% Parker Dunn
% Ec 503 Project
% 28 April 2022

% Testing "train_test_CS_IV_svm_kernel" with multiple kernels

%% Setup/preparation

clear
close all

load satimage_data.mat

[m, d]=size(Xtr);

med = median(Xtr,'omitnan');
Xtr = fillmissing(Xtr,'constant',med);
med = median(Xte,'omitnan');
Xte = fillmissing(Xte,'constant',med);

fid = fopen("results.txt",'a');

folds = 1;
classes = unique(ytr);

C = 10.^(-6:-4);

%% Validation with Linear Kernel
fprintf(fid, "\n----------\nStarting with Linear kernel\n");
tic
errors = zeros(numel(C), 1);
for i = 1:numel(C)
    % param = anything for linear kernel
    % "1" argument ==> linear kernel
    errors(i) = k_fold_cross_validation_SVM(Xtr, ytr, folds, 1, numel(classes), C(i), 0);
end
fprintf(fid, "Time to train/validate with linear kernel and THREE C parameters: %.4f", toc);


figure(1)
plot(-6:-4, 100*(1-errors), '*k')
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
fprintf(fid, "Time to test with LINEAR kernel: %.4f", toc);
fprintf(fid, "Testing Error with LINEAR kernel: %.3f", test_err);

%% Validation with Polynomial Kernel (Quadratic)
fprintf(fid, "\nStarting with Polynomial (quadratic) kernel\n");
tic
errors = zeros(numel(C), 1);
for i = 1:numel(C)
    % param = 2 -> quadratic polynomial kernel
    % "2" argument ==> polynomial kernel
    errors(i) = k_fold_cross_validation_SVM(Xtr, ytr, folds, 2, numel(classes), C(i), 2);
end
fprintf(fid, "Time to train/validate with POLYNOMIAL kernel and THREE C parameters: %0.4f", toc);

figure(2)
plot(-6:-4, 100*(1-errors), '*k')
xlabel("Exponent of C")
ylabel("Accuracy")
axis padded

[~,idx]=min(errors);
fprintf(fid, 'Best C by cross-validation with POLYNOMIAL (quad) kernel: %e\n', C(idx));

% Testing procedure for this kernel
tic
Ktrain=compute_kernel_matrix(Xtr,Xtr,2,2);
Ktest=compute_kernel_matrix(Xte,Xtr,2,2);
[test_err, alpha] = train_test_CS_IV_svm_kernel(Ktrain, Ktest, ytr, yte, numel(classes), C(idx));
fprintf(fid, "Time to test with POLYNOMIAL (quad) kernel: %.4f", toc);
fprintf(fid, "Testing Error with POLYNOMIAL (quad) kernel: %.3f", test_err);

%% Validation with RBF Kernel (aka Gaussian)


% Testing procedure for this kernel



%% close file where I am saving results
result = fclose(fid);