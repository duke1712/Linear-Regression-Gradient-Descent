function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta_temp=[zeros(size(X,2),1)];
    
    for i=1:m
        for j=1:n
            %fprintf("theta: %f\n", size(theta));
            %fprintf("X: %f\n", size(X(i,:)));
            theta_temp(j) = theta_temp(j) + ( (transpose(theta) * transpose(X(i,:)) )-y(i))*X(i,j);
    end
    theta=theta - alpha*(1/m)*theta_temp;
    % for j=1:
    % theta_2=theta(2) - alpha*(1/m)*J;
    % theta(1)=theta_1(1)
    % theta(2)=theta_2(1)









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %fprintf("Cost for %d: %f\n",iter, computeCostMulti(X,y,theta));
end

end
