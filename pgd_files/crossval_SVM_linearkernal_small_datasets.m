%% Header
% Parker Dunn
% Ec 503 Project
% 28 April 2022

% 10-fold Cross Validation for small datasets

%% Setup/preparation

clear
close all

fid = fopen("results_small_data.txt",'a');

%load iris_data.mat
%load wine_data.mat
%load glass_data.mat
%load vehicle_data.mat
%load vowel_data.mat   <-- could not get this to run
load segment_data.mat

%dataset = "iris";
%dataset = "wine";
%dataset = "glass";
%dataset = "vehicle";
%dataset = "vowel";
dataset = "segment";

date = "3 May 2022";
fprintf(fid, "\n----------\nDate: %s\nTesting Linear kernel - Dataset = %s\n", date, dataset);

[m, d]=size(Xtr);

med = median(Xtr,'omitnan');
Xtr = fillmissing(Xtr,'constant',med);

folds = 10;
classes = unique(ytr);

vals = -10:-2;

C = 10.^(vals);

%% Validation with Linear Kernel
tic
errors = zeros(numel(C), 1);
for i = 1:numel(C)
    % param = anything for linear kernel
    % "1" argument ==> linear kernel
    errors(i) = k_fold_cross_validation_SVM(Xtr, ytr, folds, 1, numel(classes), C(i), 0);
end
fprintf(fid, "Time to train/validate with linear kernel and NINE C parameters: %.4f\n", toc);


figure(1)
plot(vals, 100*(1-errors), '*k')
xlabel("C")
ylabel("Accuracy")
axis padded

[~,idx]=min(errors);
fprintf(fid, 'Best C by cross-validation with linear kernel: %e\n', C(idx));

fprintf(fid, "Cross validation error for this small dataset: %.2f%%", 100*(1-mean(errors)));