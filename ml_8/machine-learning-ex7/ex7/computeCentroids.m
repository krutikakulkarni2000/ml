function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);%m=300 %n=2


% You need to return the following variables correctly.
centroids = zeros(K, n);
%dim= 3 X 2

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
%for i=1:K
  %c=0;
  %sumx=0;
  %sumy=0;
  %for j=1:length(idx)
    %if(idx(j)==i)
    %sumx=sumx+X(j,1);
    %sumy=sumx+X(j,2);
    %c++;
    %end
  %end
  %centroids(i,1)=sumx/c;
  %centroids(i,2)=sumx/c;
%end

for i=1:K
  temp=find(idx==i);
  centroids(i,:)=mean(X(temp,:));
end





% =============================================================


end

