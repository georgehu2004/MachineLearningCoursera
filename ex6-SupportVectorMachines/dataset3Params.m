function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
% List of parameter values
param = [0.01 0,03 0.1 0.3 1 3 10 30];
% Initialize error matrix
error = zeros(length(param), length(param));
for C_index = 1:length(param)
    for sigma_index = 1:length(param)
        % Train an SVM classifier
        model = svmTrain(X, y, param(C_index), @(x1, x2)gaussianKernel(x1, x2, param(sigma_index)));
        % Predict labels for cross-validation set
        predictions = svmPredict(model, Xval);
        % Cross-validation error
        error(C_index, sigma_index) = mean(double(predictions ~= yval));
    end
end

% Find indices of the minimum cross-validation error
[C_index, sigma_index] = find(error == min(error(:)));
% Optimal C
C = param(C_index);
% Optimal sigma
sigma = param(sigma_index);

% =========================================================================

end
