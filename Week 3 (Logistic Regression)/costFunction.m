function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

grad = zeros(size(theta));
% initializing an empty matrix for future reference
h = sigmoid(X*theta);
% remember that sigmoid function uses H(x) as parameter, is equivalent to sigmoid function with X and theta

J = (1/m) * ((-y' * log(h)) - (1-y)' * log(1-h));
% plugging into cost function equation that multiplies (1/m) with the sum of the logarithmic costs of both -y log(h) or (1-y) log(1-h) depending on if y = 1 or h

grad = (1/m) * (h-y)' * (X);
% performing the necessary operations on our final output, including the usage of the partial derivative term and multiplying a transposed h-y with x


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
