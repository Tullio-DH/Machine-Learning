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


% Add ones to the X data matrix
X = [ones(m, 1) X];

% Shape each y into a matrix of ROW vectors,
% each non-null only in the k-th element
yMat = zeros(m,num_labels);

for j = 1:m
  yMat(j,y(j)) = 1;
end

%alternatively with logical operators:
%but something's wrong...
%for j = 1:m
% yMat(j,:) = 1:num_labels;
%  %logical operator:
%  yMat(j,:) == y(j);
%end

% Hidden layer (5000 x 25 elements)
z2 = X * Theta1';
a2 = sigmoid(z2);
% Add the row of bias units
a2 = [ones(m, 1) a2];

%"size of a2 is ", size(a2)

% Output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%"size of a3 is ", size(a3)

J += sum( sum( (- yMat .* log(a3) ...
      - (ones(m,num_labels)-yMat) .* log(ones(m,num_labels) - a3) ) ))...
      /m;

      
% Add regularization to J:
%One must not count the theta terms for the bias units!
%i.e., the first column of each theta matrix

regularization = ( sum(sum(Theta1(:,2:end).^2)) ...
                + sum(sum(Theta2(:,2:end).^2)) )...
                * lambda / (2*m);
                
J = J + regularization;
      

%
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

% Remove first column, the bias units coefficients
a1_nobias = X(:,2:end);
a2_nobias = a2(:,2:end);
Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);
                

% Matrix of errors on output layer
delta_3 = a3 - yMat;
%"size delta_3", size(delta_3) %5000x10

% has same dimensions as a2
% up until here the bias units are STILL included in the calculation!
delta_2 = (delta_3 * Theta2) .* ...
          [ones(size(z2,1),1) sigmoidGradient(z2)];
%"size delta_2", size(delta_2) %5000x26

% has same dimensions as Theta1
% one must not include delta(2)_0:
Theta1_grad += (delta_2(:,2:end)'*X) /m;
%"size Theta1_grad", size(Theta1_grad) %25x401

% has same dimensions as Theta2
Theta2_grad += (delta_3'*a2) /m;
%"size Theta2_grad", size(Theta2_grad) %10x26




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Note that you should not be regularizing thefirst column of Theta_grad
% which is used for the bias term
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
