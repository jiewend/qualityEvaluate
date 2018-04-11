function [J grad] = nnCostFunctionReLU(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, output_num, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
                 num_labels, (hidden_layer_size + 1));
Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


Xadd_a0 = [ones(m ,1) X];%5000 * 401
a1 = X;%5000 * 400
a1add_a0 = Xadd_a0;
z2 = Xadd_a0 * Theta1';%5000 *25
a2 = ReLU(z2);% 5000 * 25
a2add_a0 = [ones(size(a2, 1), 1) a2];% 5000 * 26
z3 = a2add_a0 * Theta2';
a3 = ReLU(z3);%5000 * 10
a3add_a0 = [ones(size(a3, 1), 1) a3];
z4 = a3add_a0 * Theta3';
a4 = z4;

Y = a4;
J = sum(sum((Y .- y).^2)) / (2 * m) + lambda * sum(nn_params.^2) / (2*m);

Delta3 = 0;
Delta2 = 0;
Delta1 = 0;
for i = 1:m;
	delta4 = a4(i,:) .- y(i,:);
	delta3 = delta4 * Theta3(:,2:end) .* ReLUGradient(z3(i,:));%1 * 10
	delta2 = delta3 * Theta2(:,2:end) .* ReLUGradient(z2(i,:));%1 * 25

	Delta3 = Delta3 + (a3add_a0(i,:)' * delta4)';
	Delta2 = Delta2 + (a2add_a0(i,:)' * delta3)';%10 * 25
	Delta1 = Delta1 + (a1add_a0(i,:)' * delta2)';%25 * 400
end,

Theta1_grad = Delta1 ./ m;%10 * 26
Theta2_grad = Delta2 ./ m;%25 * 401
Theta3_grad = Delta3 ./ m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end);
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + lambda / m * Theta3(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];



end
