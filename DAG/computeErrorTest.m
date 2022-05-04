function [accuracyRate, bestC, index] = computeErrorTest(yGuess, yTrue)

	temp1 = 2 .* (ones(1,15));
	temp2 = linspace(-2,12,15);
	C = temp1 .^ temp2;

    [a, b] = size(yGuess);
    yCorrect = zeros(a, b);
    for i = 1:b
        yCorrect(:,b) = (yGuess(:,b) == yTrue(:,1));
    end

    rates = zeros(1,b);
    for k = 1:b
        numCorrect = sum(yCorrect(:,b));
        fprintf('numCorrect = %d\n', numCorrect)
        rates(b) = numCorrect / a;
    end
    
    [accuracyRate, index] = max(rates);
    accuracyRate = accuracyRate * 100;
    bestC = C(index);
    fprintf('Best rate is %.3f with C = %.3f at index %d\n', accuracyRate, bestC, index)
    
end
