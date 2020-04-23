function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=sigmoid(X*theta);
thetasq=theta(2:size(theta));
%prints values from 2 upto size of theta
%thetasq=theta(2;size(theta));because we do not need theta0
J=sum(((-y.*log(h))-(1-y).*log(1-h)))/m +(lambda/(2*m))*sum((thetasq.^2));
thetanew=theta;
thetanew(1)=0;%do not regularize theta0
grad=(1/m)*((h.*ones(m,1)-y)'*X)+(lambda/(m)).*thetanew';





% =============================================================

end
