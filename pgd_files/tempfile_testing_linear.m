fid = fopen("results.txt",'a');

m3 = size(Xtr, 1);
m4 = size(Xte, 1);

Xtr = Xtr(mod(1:m3,3) == 0, :);
Xte = Xte(mod(1:m4,3) == 0, :);
ytr = ytr(mod(1:m3,3) == 0);
yte = yte(mod(1:m4,3) == 0);

% Testing procedure for this kernel
tic
Ktrain=compute_kernel_matrix(Xtr,Xtr,1,0);
Ktest=compute_kernel_matrix(Xte,Xtr,1,0);
[test_err, alpha] = train_test_CS_IV_svm_kernel(Ktrain, Ktest, ytr, yte, numel(classes), C(idx));
fprintf(fid, "Time to test with LINEAR kernel: %.4f\n", toc);
fprintf(fid, "Testing Error with LINEAR kernel: %.3f\n", test_err);