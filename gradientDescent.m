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







    % ============================================================

    % Save the cost J in every iteration
    % J_history(iter) = computeCost(X, y, theta);
    % grad = gradient(theta)
    % theta = theta - J_history[iter]
    prediction = X * theta;
    errorInPrediction = prediction - y;
    x_greadient = X' * errorInPrediction;
    theta = theta - (alpha * (1 / m) * x_greadient);
    J_history(iter) = sum(computeCost(X, y, theta));

end

end
