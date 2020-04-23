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

%crow=[0.01 0.03 0.1 0.3 1 3 10 30];
%sigrow=[0.01 0.03 0.1 0.3 1 3 10 30];
%result=zeros(64,3);
%sortres=zeros(64,3);
%row=1;
%for i=1:8
  %for j=1:8
    %model= svmTrain(X, y, crow(i), @(x1, x2) gaussianKernel(x1, x2, sigrow(j)));
    %predictions=svmPredict(model, Xval);
    %costpred=mean(double(predictions ~= yval));
    %result(row,1)=i;
    %result(row,2)=j;
    %result(row,3)=costpred;
    %row++;
 % end
%end
%sortres=sortrows(result,3);
%C=sortres(1,1);
%sigma=sortres(1,2);
 list  = [0.01 0.03 0.1 0.3 1 3 10 30];
 
 
 
 results = zeros(64,3);
 row=1;

for i=1:length(list)
    for j=1:length(list)
  result(row,1)=list(i);
  result(row,2)=list(j);
        model = svmTrain(X, y,list(i), @(x1, x2) gaussianKernel(x1, x2, list(j)));
        predictions = svmPredict(model, Xval);
        result(row,3) = mean(double(predictions ~= yval));
        row++;
    end
end

sorted_result = sortrows(result, 3);
C = sorted_result(1,1);
sigma = sorted_result(1,2);

% =========================================================================

end
