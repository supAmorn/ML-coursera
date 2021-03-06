function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(size(X,2)); %number theta
iter = size(theta,1); %number of iteration
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%calculate sigmoid 
  g = sigmoid(X*theta);
%Compute cost 
  var = -y.*log(g)-(1-y).*log(1-g);
  J = 1/m*sum(var);
% Compute Gradient
  for i = 1:iter
    tmp(i) = (1/m)*sum((g-y).*X(:,i));
  endfor
  grad = tmp;

% =============================================================

end
