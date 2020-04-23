function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X=12X2
%y=12X1
%theta=2X1
%grad=2X1

hx=X*theta;%12X1
J=(1/(2*m))*sum((hx - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);
%gradient
grad(1) = (1/m)*(X(:,1)'*(hx-y)); % scalar == 1x1
%X(:,1)=12X1, transpose 1X12
grad(2:end) = (1/m)*(X(:,2:end)'*(hx-y)) + (lambda/m)*theta(2:end);
% =========================================================================

grad = grad(:);

end
