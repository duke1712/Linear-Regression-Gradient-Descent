function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    J=0;
    for i=1:m
        J = J + ( (transpose(theta) * transpose(X(i,:)) )-y(i));
    end
    theta_1=theta(1) - alpha*(1/m)*J;
    J=0;
    for i=1:m
        J = J + ( (transpose(theta) * transpose(X(i,:)) )-y(i))*X(i,2);
    end
    theta_2=theta(2) - alpha*(1/m)*J;
    theta(1)=theta_1(1)
    theta(2)=theta_2(1)

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
