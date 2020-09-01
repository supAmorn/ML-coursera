function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%



%set up X dimension
  X = [ones(m,1),X];
 
%calculate hidden layer 1
  z2 = X*Theta1'
  a2 = sigmoid(z2);

%calculate output
  a2 = [ones(m, 1) a2];
  h_theta = sigmoid(a2*Theta2');

%Set up new y output as 1*10 for each result
  
%Set up Y as a matrix training set * number of label(K)
  Yk = zeros(m,num_labels);
##  Y = zeros(m,num_labels);
##  for j =1:m
##    %set up y as a matrix 1*number of label(K)
##    y_row = zeros(1,num_labels);
##    y_row(y(j)) =1; %assign output as 1 into a column for each row
##    Y(j,:) = y_row;
##  endfor

  for j = 1:m
    Yk(j,y(j)) =1; %assign output as 1 into a column for each row
  endfor

##%Compute cost J for each number of label
##  J_part = zeros(1,num_labels);
##  for i = 1:num_labels
##    var = -Y(:,i).*log(h_theta(:,i))-(1-Y(:,i)).*log(1-h_theta(:,i));
##    J_part(:,i) = 1/m*sum(var);
##  endfor
##  
##  J = sum(J_part(:))

   J = 1/m*sum(sum(-Yk.*log(h_theta)-(1-Yk).*log(1-h_theta)))
  

% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% =========================================================================

  

%delta level 3  
  D3 = h_theta - Yk;
%delta level 2
  D2 = (D3*Theta2)(:,2:end).*sigmoidGradient(z2);
  
  Theta1_grad = Theta1_grad + (1/m).*(D2'*X);
  Theta2_grad = Theta2_grad + (1/m).*(D3'*a2);

  
   
% -------------------------------------------------------------
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% =========================================================================
% Reshape Theta1 and Theta2 (Exclude 1st both theta for regularization)
Theta1_reg = reshape(nn_params((hidden_layer_size+1):hidden_layer_size * (input_layer_size+1)), ...
                 hidden_layer_size, (input_layer_size));
Theta2_reg = reshape(nn_params((1+num_labels+ (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size));


%include regularization cost
cost = (lambda/(2*m))*(sum(Theta1_reg(:).^2)+sum(Theta2_reg(:).^2))                
J = J+cost

%partial derivatives Theta
reg1 = (lambda/m)*Theta1(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) +reg1;

reg2 = (lambda/m)*Theta2(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) +reg2;

% -------------------------------------------------------------
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
