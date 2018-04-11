function [X_train, y_train, X_val, y_val, X_testA, X_testB, initial_nn_params] = get_X_and_y();
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

