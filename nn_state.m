clear ; close all; clc; 

load('afterPAC_DataSet304.mat');
trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;


X = afterPAC_DataSet(1:trainSetNum, :);

sumK = size(X, 2);
input_layer_size  = sumK
hidden_layer_size = sumK
num_labels = 1
output_num = 1

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Theta3 = randInitializeWeights(num_labels, output_num);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

iter_num = 1000;
options = optimset('MaxIter', iter_num);
lambda = 0.25;
% pred_train = zeros(400, length(lambda));
% pred_val = zeros(100, length(lambda));

trainSetNums = [305:5:400];
error_train = zeros(size(trainSetNums));
error_val = zeros(size(trainSetNums));
Thetas = zeros(length(trainSetNums), length(initial_nn_params));

j = 0;

L = length(trainSetNums);
fprintf('============ begin at %s ============\n', datestr(now));
for i = trainSetNums,
	j++;
	randSort = randperm(size(X, 1));
	X = X(randSort, :);
	y = y(randSort, :);

	X_train = X(1:i, :);
	y_train = y(1:i, :);
	X_val = X(i+1:i+100, :);
	y_val = y(i+1:i+100, :);

	costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, output_num, X_train, y_train, lambda);
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
					 num_labels, (hidden_layer_size + 1));
	Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));

	Thetas(j, :) = nn_params;
	pred_train = predict_3_layer(Theta1, Theta2, Theta3, X_train);
	pred_val = predict_3_layer(Theta1, Theta2, Theta3, X_val);

	error_train(j) = sum((pred_train .- y_train).^2) / i;
	error_val(j) = sum((pred_val .- y_val).^2) / 100;

	fprintf('%d%% ======== trainNum : %d error_train : %f error_val : %f ==========\n', j/L * 100, i, error_train(j), error_val(j))
	clear pred_train pred_val;

end,

plot(trainSetNums, error_train, 'r', trainSetNums, error_val, 'g');
legend('errorTrain', 'errorVal');
fprintf('============ end at %s ============\n', datestr(now));

