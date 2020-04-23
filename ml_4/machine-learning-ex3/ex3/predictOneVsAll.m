function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%dim of all_theta are 10 X 401 , thus K has value=1
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);% m=5000
num_labels = size(all_theta, 1);%num_labels=10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);% p is a zeros matrix of dim 5000 X 1

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.

%       X=5000 X 401
%       all_theta= 10 X 401

pred = X*all_theta';% predictions
% pred = 5000 X 10
[probability,p]=max(pred,[],2);
%[probability,p] dim= 2X1
%probability and p are vectors
%probability contains the max value in the row (the actual
%probability) dim=5000X1
%p contains the indices of the max values of pred dim=5000X1
%now lets see how max works with eg
% A=[1 5 3; 8 6 3]
%A =
%
 %  1   5   3
  % 8   6   3
%
%octave:23> [a,b]=max(A,[],2)
%a =
%
 %  5
  % 8
%
%b =
%
 %  2
  % 1

% =========================================================================


end