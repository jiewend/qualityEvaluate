clear ; close all; clc; 

pkg load statistics
load('afterPAC_DataSet450.mat');
trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;


X = afterPAC_DataSet(1:trainSetNum, :);

sumK = size(X, 2);
input_layer_size  = sumK
hidden_layer_size = 100
num_labels = 10 
output_num = 1

iter_num = 1000;
options = optimset('MaxIter', iter_num);

time_for_same_lambda = 5;
lambdas = [0:0.1:1];

fprintf('============ begin at %s ============\n', datestr(now));
for i = 1:length(lambdas),

	for j = 1:time_for_same_lambda,
		randSort = randperm(size(X, 1));
		X = X(randSort, :);
		y = y(randSort, :);

		X_train = X(1:400, :);
		y_train = y(1:400, :);
		X_val = X(401:500, :);
		y_val = y(401:500, :);

		initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
		initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
		initial_Theta3 = randInitializeWeights(num_labels, output_num);
		initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

		costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, output_num, X_train, y_train, lambdas(i));
		[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
						 hidden_layer_size, (input_layer_size + 1));
		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
						 num_labels, (hidden_layer_size + 1));
		Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));

		initial_nn_paramss(i, j, :) = initial_nn_params;
		Thetas(i, j, :) = nn_params;
		pred_train(i, j, :) = predict_3_layer(Theta1, Theta2, Theta3, X_train);
		pred_val(i, j, :) = predict_3_layer(Theta1, Theta2, Theta3, X_val);

		error_train(i, j) = sum((pred_train(i, j) .- y_train).^2) / 400;
		error_val(i, j) = sum((pred_val(i, j) .- y_val).^2) / 100;

		fprintf('========lambda:%f time: %d error_train: %f error_val: %f\n', lambdas(i), j, error_train(i, j), error_val(i, j));

	end
	locate = find(error_train > 0.06);
	error_val(locate) = NaN;
	error_train(locate) = NaN;
	fprintf('>>>>>>>>>>>lambda: %f mean_err_train: %f mean_err_val: %f <<<<<<<<<<<<\n\n', lambdas(i), nanmean(error_train(i, :)), nanmean(error_val(i, :)));
end


fprintf('=========== end at %s =========\n', datestr(now));
plot(lambdas, nanmean(error_train, 2), lambdas, nanmean(error_val, 2));
legend('errorTrain', 'errorVal');

error_diff = abs(error_val .- error_train);
[ii, jj] = find(error_diff < 0.003);

fprintf('========== hand pick ========\n');
for k = 1:length(ii),
	if error_val(ii(k), jj(k)) < 0.05,
		fprintf('lambda:%f times:%d error_train : %f, error_val : %f \n' , lambdas(ii(k)), jj(k), error_train(ii(k), jj(k)), error_val(ii(k), jj(k))); 
	end
end



