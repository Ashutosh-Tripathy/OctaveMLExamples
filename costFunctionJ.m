% X = [1 1; 1 2; 1 3];  y = [1; 2; 3]; theta = [0;1]; costFunctionJ(X, y, theta)
function J = costFunctionJ(X, y, theta)

m = size(X, 1)
predictions = X * theta
sqrErrors = (predictions - y) .^ 2

J = 1 / (2 * m) * sum(sqrErrors)