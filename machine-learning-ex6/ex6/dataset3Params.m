function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = dataset3Params(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
[CValues, sigmaValues] = meshgrid(values, values);
combinations = [CValues(:), sigmaValues(:)];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
minError = realmax;
for i = 1:size(combinations, 1)
    currC = combinations(i, 1);
    currSigma = combinations(i, 2);

    fprintf('Training SVM with rbf kernel with parameters: C = %f, sigma = %f.\n',
            currC, currSigma);

    model = svmTrain(X, y, currC,
                     @(x1, x2) gaussianKernel(x1, x2, currSigma));

    fprintf('Cross-validating SVM with rbf kernel with parameters: C = %f, sigma = %f.\n',
            currC, currSigma);

    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));

    fprintf('Minimum error: %f\nCurrent model error: %f.\n', minError, error);

    if error < minError
        fprintf('Current model is better than the others.\n');

        minError = error;
        C = currC;
        sigma = currSigma;
    end

    fprintf('Current best parameters: C = %f, sigma = %f.\n\n', C, sigma);
end
% =========================================================================

end
