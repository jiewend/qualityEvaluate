function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
for i = 1:length(sigma),
	if sigma(i) != 0,
		X_norm(:, i) = bsxfun(@rdivide, X_norm(:,i), sigma(i));
	end
end


% ============================================================

end