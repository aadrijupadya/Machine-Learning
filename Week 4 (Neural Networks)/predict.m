function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

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
X = [ones(m,1) X];
% Creating a column of ones to account for bias unit
a2 = sigmoid(X * Theta1');
% Calculating a2 for next layer of propagation by taking sigmoid function of the product of X and Theta1 (weights created by input layer)

a2 = [ones(m,1) a2];
% adding another column of ones to account for bias unit 
classifier = sigmoid(a2 * Theta2');
% taking sigmoid function of a2 times theta2 to find a3 or the classifier in this case

[~,p] = max(classifier, [], 2);
% final prediction will be an array that pulls the labels with the highest output functions








% =========================================================================


end
