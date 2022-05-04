function [results, tValEnd] = justValTest(Xval, first, second, newC)
	numClasses = 6;
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
    
    mVal = size(Xval,1);
    results = zeros(mVal,span);

    tValStart = tic;
    for iterC = first:second
        %fprintf('C = %d\n', C(iterC));
        for k = 1:mVal
            %fprintf('m = %d\n', k);
            start = 1;
            fin = numClasses;
            index = indexMap(start,fin);
            sample = Xval(k,:);
            weights = newC{iterC}{index};
            [sm, sd] = size(sample);
            [wm, wd] = size(weights);
            %fprintf('sample is %d by %d, weights is %d by %d\n', sm, sd, wm, wd);
            prediction = testSVMTest(sample, weights);
            while (start ~= fin)
                if (prediction > 0)
                    prediction = start;
                    fin = fin - 1;
                    if ((start == fin) || (indexMap(start,fin) == 0))
                        break;
                    end
                    index = indexMap(start,fin);
                    sample = Xval(k,:);
                    weights = newC{iterC}{index};
                    [sm, sd] = size(sample);
                    [wm, wd] = size(weights);
                    %fprintf('sample is %d by %d, weights is %d by %d\n', sm, sd, wm, wd);
                    prediction = testSVMTest(sample, weights);
                else
                    prediction = fin;
                    start = start + 1;
                    if ((start == fin) || (indexMap(start,fin) == 0))
                        break;
                    end
                    index = indexMap(start,fin);
                    sample = Xval(k,:);
                    weights = newC{iterC}{index};
                    [sm, sd] = size(sample);
                    [wm, wd] = size(weights);
                    %fprintf('sample is %d by %d, weights is %d by %d\n', sm, sd, wm, wd);
                    prediction = testSVMTest(sample, weights);
                end
            end
            results(k, iterC) = prediction;
        end
    end
    tValEnd = toc(tValStart);

    uniqueTotal = unique(results);
    fprintf('unique values = %d\n', uniqueTotal);

end