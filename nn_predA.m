clear ; close all; clc; 

load('afterPAC_DataSet450.mat');
trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;

X = afterPAC_DataSet(1:trainSetNum, :);
X_train = X;
y_train = y;

X_testA = afterPAC_DataSet(501:600, :);

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

costFunction = @(p) nnCostFunctionReLU(p, input_layer_size, hidden_layer_size, num_labels, output_num, X_train, y_train, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

y_A = predict_3_layer1(nn_params, 450, 450, 1, 1, X_testA);

