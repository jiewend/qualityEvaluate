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
X_testA = afterPAC_DataSet(trainSetNum + 1:trainSetNum+testASetNum, :);
X_testB = afterPAC_DataSet(trainSetNum + testASetNum + 1:trainSetNum + testASetNum + testBSetNum,:);
sumK = size(X, 2) - 13;
input_layer_size  = sumK + 13 
hidden_layer_size = 400
% num_labels = 1;             
num_labels = 1;
output_num = 1;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Theta3 = randInitializeWeights(num_labels, output_num);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
iter_num = 10;
options = optimset('MaxIter', iter_num);
lambda = 0.05;

nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, output_num, X, y, lambda);

J
fprintf('lambda = %f num_labels = %f iter_num = %f \n',lambda, num_labels, iter_num);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, output_num, X, y, lambda);
% options = optimset('MaxIter', 500, 'GradObj', 'on');
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size + 1) * num_labels)), ...
                 num_labels, (hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + numel(Theta1)) + numel(Theta2):end), output_num, (num_labels + 1));



pred = predict_3_layer(Theta1, Theta2, Theta3, X);

predA = predict_3_layer(Theta1, Theta2, Theta3, X_testA);

cost = sum((pred .- y).^2) / 500;
fprintf('cost = %f  lambda = %f num_labels = %f iter_num = %f \n',cost, lambda, num_labels, iter_num);

fprintf('============ end at %s ============\n', datestr(now));
