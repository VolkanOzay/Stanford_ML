function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


positive = log(sigmoid(X*theta));
negative = log(1-sigmoid(X*theta));
regterm1 = (lambda/(2*m))*sum(theta(2:length(theta),1).^2);
J = (1/m)*(sum(-y.*positive - (1.-y).*negative)) + regterm1;
regterm2 = (lambda/m)*theta(2:length(theta));
grad = ((1/m) * (sum(X.*(sigmoid(X*theta).-y))))' .+ [0 ;regterm2];
fprintf('%f',grad);



% =============================================================

end
