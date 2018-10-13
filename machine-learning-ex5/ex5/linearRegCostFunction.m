function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h=X*theta;
J = sum((h-y).^2);
J+=lambda*sum(theta(2:end).*theta(2:end));
J/=2*m;

reg = [0;theta(2:end)]'.*(lambda/m);
t = (h-y).*X; % h-y is multiplied to each column in X
t=sum(t); % flat row vector with same columns as X % regularise
t/=m; % divide by m

grad = t+reg;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
