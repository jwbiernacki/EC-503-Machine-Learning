function [accuracyRate] = computeErrorV2(yGuess, yTrue)

    [a, ~] = size(yGuess);
	yCorrect = (yGuess == yTrue);

	numCorrect = sum(yCorrect);
	fprintf('numCorrect = %d\n', numCorrect)
    accuracyRate = (numCorrect / a) * 100;
    
    fprintf('Accuracy rate is %.3f\n', accuracyRate)
    
end
