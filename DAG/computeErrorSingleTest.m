function [accuracyRate, bestC] = computeErrorTest(yGuess, yTrue)

	temp1 = 2 .* (ones(1,15));
	temp2 = linspace(-2,12,15);
	C = temp1 .^ temp2;

    [a, b] = size(yGuess);
    yCorrect = zeros(a, b);
    for i = 1:a
        for j = 1:b
        yCorrect(a,b) = (yGuess(a,b) == yTrue(a,1));
        end
    end

    rates = zeros(1,b);
    for k = 1:b
        rates(b) = sum(yCorrect(:,b)) / a;
    end
    
    [accuracyRate, index] = max(rates);
    bestC = C(index);
    fprintf('Best rate is %.3f with C%d = %.3f', accuracyRate, index, bestC)
    
end
