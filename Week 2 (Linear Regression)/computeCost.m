function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%     
% You should set J to the cost.
H = X * theta;
% Matrix that is product of X and theta
J = 1/(2*m) * sum((H-y) .^ 2);
% Cost function, 1/2m * sum(each predicted value - actual value) ^2



% =========================================================================

end
