function p = predict_3_layer1(nn_params, input_layer_size, hidden_layer_size, num_labels, output_num, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
				 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
				 num_labels, (hidden_layer_size + 1));
Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));


m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
% h1 = ReLU([ones(m, 1) X] * Theta1');
% h2 = ReLU([ones(m, 1) h1] * Theta2');
h3 = [ones(m, 1) h2] * Theta3';
% [dummy, p] = max(h2, [], 2);
p = h3;
% =========================================================================


end
