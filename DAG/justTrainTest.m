function [newC, tTrainEnd] = justTrainTest(xTrMain, yTrMain, first, second)
	numClasses = 6;
	numNodes = (numClasses * (numClasses - 1)) / 2;
	temp1 = 2 .* (ones(1,15));
	temp2 = linspace(-2,12,15);
	C = temp1 .^ temp2;

	indexMap = zeros(6,6);
	tempIndex = 0;
	for i = 1:5
        for j = (i+1):6
			tempIndex = tempIndex  + 1;
			indexMap(i,j) = tempIndex;
        end
	end
    
	span = (second - first) + 1;
    
    newC = cell(1,span);
    for i = 1:span
        newC{i} = cell(1,numNodes);
    end
    
    tTrainStart = tic;
    for iterC = first:second
        fprintf('Starting C = %d\n', C(iterC));
        for i = 1:(numClasses-1)
            for j = (i+1):(numClasses)
                index = indexMap(i,j);
                fprintf('Index %d-%d\n', iterC, index );
                weights = trainPrimalTest(yTrMain{i}, yTrMain{j}, xTrMain{i}, xTrMain{j}, C(iterC));
                newC{iterC}{index} = weights;
            end
        end
    end
    tTrainEnd = toc(tTrainStart);
    
end