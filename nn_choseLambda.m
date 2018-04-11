clear ; close all; clc; 

fprintf('============ begin at %s ============\n', datestr(now));
load('x.mat');
load('y.mat');
load('afterPAC_DataSet.mat');
trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;
fprintf('=============== training neural network =============\n');
% checkNNGradients(1);
X = afterPAC_DataSet(1:trainSetNum, :);
X_train = X(1:400, :);
X_val = X(401:500, :);
y_train = y(1:400, :);
y_val = y(401:500, :);
X_testA = afterPAC_DataSet(trainSetNum + 1:trainSetNum+testASetNum, :);
X_testB = afterPAC_DataSet(trainSetNum + testASetNum + 1:trainSetNum + testASetNum + testBSetNum,:);
sumK = size(X, 2) - 13;
input_layer_size  = sumK + 13 
hidden_layer_size = sumK + 13
num_labels = 1
output_num = 1

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Theta3 = randInitializeWeights(num_labels, output_num);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

iter_num = 1000;
options = optimset('MaxIter', iter_num);
lambda = [0.05:0.05:0.2];
pred_train = zeros(400, length(lambda));
pred_val = zeros(100, length(lambda));

error_train = zeros(size(lambda));
error_val = zeros(size(lambda));
scores = zeros(size(lambda));

predA = zeros(100, length(lambda));
Thetas = zeros(length(lambda), length(initial_nn_params));

for i = 1:length(lambda),

	costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, output_num, X_train, y_train, lambda(i));

	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
					 num_labels, (hidden_layer_size + 1));
	Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));

	Thetas(i, :) = nn_params;
	pred_train(:, i) = predict_3_layer(Theta1, Theta2, Theta3, X_train);
	pred_val(:, i) = predict_3_layer(Theta1, Theta2, Theta3, X_val);

	error_train(i) = sum((pred_train(:, i) .- y_train).^2) / 500;
	error_val(i) = sum((pred_val(:, i) .- y_val).^2) / 100;

	scores(i) = (error_train(i) .- error_val(i)).^2;
	
	fprintf('============ done lambd : %f at %s score : %f MSE_train : %f MSE_val : %f ============\n', lambda(i), datestr(now), scores(i), error_train(i), error_val(i));
end,

scores = (error_train .- error_val).^2;

fprintf('============ end at %s ============\n', datestr(now));
