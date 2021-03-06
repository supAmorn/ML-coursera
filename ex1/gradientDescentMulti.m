function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);
tmp = zeros(n,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    J = X*theta-y;
    %temp1 = theta(1) - alpha*(1/m)*sum(J.*X(:,1));
    %temp2 = theta(2) - alpha*(1/m)*sum(J.*X(:,2));
    %temp3 = theta(3) - alpha*(1/m)*sum(J.*X(:,3));
    %theta(1) = temp1;
    %theta(2) = temp2;
    %theta(3) = temp3;
    for i = 1:n,
      tmp(i) = theta(i) - alpha*(1/m)*sum(J.*X(:,i));
    endfor
    theta = tmp;









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
