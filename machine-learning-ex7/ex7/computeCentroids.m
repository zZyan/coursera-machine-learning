function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
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
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.

% idx: have the matrix of centroids assignment

% counter=zeros(K,1);
% for i=1: m
%     centroids(idx(i),:)=centroids(idx(i),:)+X(i,:);
%     counter(idx(i))=counter(idx(i))+1;
% end
% display(centroids);
% for j=1:n
%     centroids(:,j)=centroids(:,j)./counter;
% end
 
for i=1:K
    k_vector=idx==i;
    k_counter=sum(k_vector);
    k_matrix=repmat(k_vector,1,n);
    k_sum=sum(k_matrix.*X); %sum by default sum up column
    centroids(i,:)=k_sum./k_counter;
%     display(size(k_sum));
%     display(size(k_counter));
end

end

