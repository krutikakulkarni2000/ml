function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);%m=5000
%how size works
%m=size(X,1)
%m =  5000
num_labels = size(Theta2, 1);%equals to 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%dim p= 5000 X 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%input layer=a1, adding one extra column of ones inordere to
%geet the 'bias unit'
a1=[ones(m,1) X];
%dim 5000 X 401

%Theta1 25 X 401
z2=a1*Theta1';%dim=5000 X 25
a2=sigmoid(z2);%dim=5000 X 25

%Theta2 10 X 26
a2=[ones(size(a2,1),1) a2];%dim 5000 X 26
z3=a2*Theta2';%dim 5000 X 10
a3=sigmoid(z3);%dim 5000 X 10

[probability,p]=max(a3,[],2);
% =========================================================================


end
