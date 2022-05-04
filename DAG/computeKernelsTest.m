function [kernelsOut, tKernelEnd] = computeKernelsTest(X, y)

numClasses = size(y,2);
numNodes = (numClasses * (numClasses - 1)) / 2;

Xkern = cell(1,numNodes);

% The index map is simply a tool for dealing with the structure of the graph/problem.
indexMap = zeros(numClasses);
tempIdx = 0;
for i = 1:(numClasses - 1)
	for j = (i+1):numClasses
		tempIdx = tempIdx + 1;
		indexMap(i,j) = tempIdx;
	end
end

    tKernelStart = tic;
    for i = 1:(numClasses - 1)
        for j = (i+1):numClasses
            index = indexMap(i,j);
            kernelized_Xtrain{index}{2} = X * X';


tKernelEnd = toc(tKernelStart);

for i = 1:(numClasses - 1)
	for j = (i+1):numClasses
        index = indexMap(i,j);
        kernelized_Xtrain{index}{5} = [yTrMain{i};yTrMain{j}]; 
        kernelized_Xval{index}{5} = [yValMain{i};yValMain{j}];
        kernelized_Xtest{index}{5} = [yTeMain{i};yTeMain{j}];
	end
end

kernelsOut = {kernelized_Xtrain, kernelized_Xval, kernelized_Xval};

end