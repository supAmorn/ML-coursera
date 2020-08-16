function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
iter = size(theta,1); %number of iteration
l = length(theta); %length of theta
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%calculate sigmoid 
  g = sigmoid(X*theta);
%Compute cost 
  var = -y.*log(g)-(1-y).*log(1-g);
  J = 1/m*sum(var);
%Compute regularization
  J = J+(lambda/(2*m))*sum(theta(2:l).^2);
  

% Compute Gradient(only theta 0)
  grad(1) = (1/m)*sum((g-y).*X(:,1));
% Compute Grdient >=1 (since theta 1)
  for i = 2:iter
    grad(i) = ((1/m)*sum((g-y).*X(:,i)))+(lambda/m)*theta(i);
  endfor
   


% =============================================================

end
