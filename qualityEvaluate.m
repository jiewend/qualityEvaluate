clear ; close all; clc;

pkg load statistics;
load('data_processed.mat');
load('testA.mat');
load('testB.mat');

%================deal with NaN in data===========
fprintf('============ load data with value NaN=========\n');
fprintf('========== data preprocess =========\n');
trainSetNum = size(data_processed, 1);
testASetNum = size(testA, 1);
testBSetNum = size(testB, 1);
y = data_processed(:, end);
data_processed = [data_processed(:, 1:end - 1); testA; testB];

[nanR, nanC]=find(isnan(data_processed));
data_fix_nan = data_processed;
process_name_single_letter = [1, 232, 752, 774, 945, 2382, 3894, 4169, 5978, 6575];
process_name_num = [1702, 3694, 6192];

h=0;
k=0;
a = [];
uniqueNanC = unique(nanC);
size(uniqueNanC)
for i = 1:length(uniqueNanC),
	if nansum(data_fix_nan(:, uniqueNanC(i))) == 0;
		data_fix_nan(:, uniqueNanC(i)) = 0;
		h++;
		a = [a, uniqueNanC(i)];
	else,
		[temp1, temp2] = find(isnan(data_fix_nan(:, uniqueNanC(i))));
		data_fix_nan(temp1, uniqueNanC(i)) = nanmean(data_fix_nan(:, uniqueNanC(i)));
		k++;
	end,
end,
nanNum = sum(sum(isnan(data_fix_nan)));
InfNum = sum(sum(isinf(data_fix_nan)));
fprintf('========== done with NaN value nanNum = %d   InfNum = %d========= \n',nanNum, InfNum);
pause;
%===============Dimensionality reduction========
fprintf('========== Dimensionality reduction going =========\n');
processNum1 = data_fix_nan(:, 2:231);%232
processNum2 = data_fix_nan(:, 233:751);%752
processNum3 = data_fix_nan(:, 753:773);%774
processNum4 = data_fix_nan(:, 775:944);%945
processNum5 = data_fix_nan(:, 946:1701);%1702
processNum6 = data_fix_nan(:, 1703:2381);%2382
processNum7 = data_fix_nan(:, 2383:3693);%3694
processNum8 = data_fix_nan(:, 3695:3893);%3894
processNum9 = data_fix_nan(:, 3895:4168);%4169
processNum10 = data_fix_nan(:, 4170:5977);%5978
processNum11 = data_fix_nan(:, 5979:6191);%6192
processNum12 = data_fix_nan(:, 6193:6574);%6575
processNum13 = data_fix_nan(:, 6576:8027);

[normProcessNum1, mu1, sigma1] = featureNormalize(processNum1);
[normProcessNum2, mu2, sigma2] = featureNormalize(processNum2);
[normProcessNum3, mu3, sigma3] = featureNormalize(processNum3);
[normProcessNum4, mu4, sigma4] = featureNormalize(processNum4);
[normProcessNum5, mu5, sigma5] = featureNormalize(processNum5);
[normProcessNum6, mu6, sigma6] = featureNormalize(processNum6);
[normProcessNum7, mu7, sigma7] = featureNormalize(processNum7);
[normProcessNum8, mu8, sigma8] = featureNormalize(processNum8);
[normProcessNum9, mu9, sigma9] = featureNormalize(processNum9);
[normProcessNum10, mu10, sigma10] = featureNormalize(processNum10);
[normProcessNum11, mu11, sigma11] = featureNormalize(processNum11);
[normProcessNum12, mu12, sigma12] = featureNormalize(processNum12);
[normProcessNum13, mu13, sigma13] = featureNormalize(processNum13);

[U1, S1, K1] = pca(normProcessNum1);
[U2, S2, K2] = pca(normProcessNum2);
[U3, S3, K3] = pca(normProcessNum3);
[U4, S4, K4] = pca(normProcessNum4);
[U5, S3, K5] = pca(normProcessNum5);
[U6, S6, K6] = pca(normProcessNum6);
[U7, S7, K7] = pca(normProcessNum7);
[U8, S8, K8] = pca(normProcessNum8);
[U9, S9, K9] = pca(normProcessNum9);
[U10, S10, K10] = pca(normProcessNum10);
[U11, S11, K11] = pca(normProcessNum11);
[U12, S12, K12] = pca(normProcessNum12);
[U13, S13, K13] = pca(normProcessNum13);

reducedProcess1 = projectData(normProcessNum1, U1, K1);
reducedProcess2 = projectData(normProcessNum2, U2, K2);
reducedProcess3 = projectData(normProcessNum3, U3, K3);
reducedProcess4 = projectData(normProcessNum4, U4, K4);
reducedProcess5 = projectData(normProcessNum5, U5, K5);
reducedProcess6 = projectData(normProcessNum6, U6, K6);
reducedProcess7 = projectData(normProcessNum7, U7, K7);
reducedProcess8 = projectData(normProcessNum8, U8, K8);
reducedProcess9 = projectData(normProcessNum9, U9, K9);
reducedProcess10 = projectData(normProcessNum10, U10, K10);
reducedProcess11 = projectData(normProcessNum11, U11, K11);
reducedProcess12 = projectData(normProcessNum12, U12, K12);
reducedProcess13 = projectData(normProcessNum13, U13, K13);

sumK = (K1 + K2 + K3 +K4 + K5 +K6 +K7 +K8 + K9 + K10 + K11 + K12 + K13)


afterPAC_DataSet = [data_fix_nan(:, 1), reducedProcess1, data_fix_nan(:, 232), reducedProcess2, data_fix_nan(:, 752), reducedProcess3, data_fix_nan(:, 774), reducedProcess4, data_fix_nan(:, 945), reducedProcess5, data_fix_nan(:, 1702), reducedProcess6, data_fix_nan(:, 2382), reducedProcess7, data_fix_nan(:, 3694), reducedProcess8, data_fix_nan(:, 3894), reducedProcess9, data_fix_nan(:, 4169), reducedProcess10, data_fix_nan(:, 5978), reducedProcess11, data_fix_nan(:, 6192), reducedProcess12, data_fix_nan(:, 6575), reducedProcess13];


fprintf('======== done with Dimensionality reduction, remain Dimensionality : %d\n', sumK+13);
fprintf('================= done wtih data preprocess===========\n')
pause;
pause;

load('x.mat');
load('y.mat');
load('afterPAC_DataSet.mat');
fprintf('=============== training neural network =============\n');
% checkNNGradients(1);
X = afterPAC_DataSet(1:trainSetNum, :);
X_testA = afterPAC_DataSet(trainSetNum + 1:trainSetNum+testASetNum, :);
X_testB = afterPAC_DataSet(trainSetNum + testASetNum + 1:trainSetNum + testASetNum + testBSetNum,:);
sumK = size(X, 2) - 13;
input_layer_size  = sumK + 13  
hidden_layer_size = sumK + 13  
% num_labels = 1;             
num_labels = 500;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

iter_num = 500;
options = optimset('MaxIter', iter_num);
lambda = 0.05;

nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
                   
[norm_y plus_y time_y] = my_norm(y);
[discreteY grad_y min_y] = discrete_y(y, num_labels);
J = t_nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, discreteY, lambda);
J
fprintf('lambda = %f num_labels = %f iter_num = %f \n',lambda, num_labels, iter_num);
% costFunction = @(p) nnCostFunction(p, ...
                                   % input_layer_size, ...
                                   % hidden_layer_size, ...
                                   % num_labels, X, norm_y, lambda);
costFunction = @(p) t_nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, discreteY, lambda);
								   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% pred = (predict(Theta1, Theta2, X) + plus_y) * time_y;
pred = concrete_y(t_predict(Theta1, Theta2, X), grad_y, min_y);
% predA = (predict(Theta1, Theta2, X_testA) + plus_y) * time_y;
predA = concrete_y(t_predict(Theta1, Theta2, X_testA), grad_y, min_y);
cost = sum((pred .- y).^2) / 500;
fprintf('cost = %f  lambda = %f num_labels = %f iter_num = %f \n',cost, lambda, num_labels, iter_num);












