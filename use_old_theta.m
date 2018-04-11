clear ; close all; clc; 

load('afterPAC_DataSet450.mat');
% load('multi_lambda_multi_time.mat')
load('nn_multi_init4.mat');

trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;

X = afterPAC_DataSet(1:trainSetNum, :);

X_train = X(1:500, :);
y_train = y;
X_testA = afterPAC_DataSet(501:600, :);
sumK = size(X, 2);
input_layer_size  = sumK
hidden_layer_size = 100
num_labels = 10
output_num = 1

iter_num = 3000;
options = optimset('MaxIter', iter_num);

% initial_nn_params1 = Thetas(10, 5,:);
initial_nn_params1 = initial_nn_paramss(10, 5,:);
initial_nn_params = initial_nn_params1(:);
size(initial_nn_params)


costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, output_num, X_train, y_train, 0.9);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% predA = predict_3_layer1(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, output_num, X_testA);
pred_train = predict_3_layer1(nn_params, input_layer_size, hidden_layer_size, num_labels, output_num, X_train);
predA = predict_3_layer1(nn_params, input_layer_size, hidden_layer_size, num_labels, output_num, X_testA);
error_train = sum((pred_train .- y).^2) / 500

predA
