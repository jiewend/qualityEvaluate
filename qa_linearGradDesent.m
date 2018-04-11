clear; close all; clc;

% pkg load statistics;
load('afterPAC_DataSet');
load('y.mat');

trainSetNum = 500;
testASetNum = 100;
testBSetNum = 121;
X = afterPAC_DataSet(1:trainSetNum, :);
X_testA = afterPAC_DataSet(trainSetNum + 1:trainSetNum+testASetNum, :);
X_testB = afterPAC_DataSet(trainSetNum + testASetNum + 1:trainSetNum + testASetNum + testBSetNum,:);

% J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

Xval = X(401:500, :);
yval = y(401:500, :);

Xtest = X(401:500, :);
ytest = y(401:500, :);

X = X(1:400, :);
y = y(1:400, :);

lambda = 0.1;
[error_train, error_val] = learningCurve(X, y, Xval, yval, 0.1, 0.1, 10);
    
plot(1:m, error_train, 1:m, error_val);
