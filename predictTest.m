
clear ; close all; clc;

pkg load io;
pkg load statistics;
load('testA.mat');

fprintf('========== data preprocess =========\n');

testA_processed = testA;
[nanR, nanC]=find(isnan(testA_processed));
testA_fix_nan = testA_processed;
process_name_single_letter = [1, 232, 752, 774, 945, 2382, 3894, 4169, 5978, 6575];
process_name_num = [1702, 3694, 6192];

h=0;
k=0;
a = [];
uniqueNanC = unique(nanC);
size(uniqueNanC)
for i = 1:length(uniqueNanC),
	if nansum(testA_fix_nan(:, uniqueNanC(i))) == 0;
		testA_fix_nan(:, uniqueNanC(i)) = 0;
		h++;
		a = [a, uniqueNanC(i)];
	else,
		[temp1, temp2] = find(isnan(testA_fix_nan(:, uniqueNanC(i))));
		testA_fix_nan(temp1, uniqueNanC(i)) = nanmean(testA_fix_nan(:, uniqueNanC(i)));
		k++;
	end,
end,
nanNum = sum(sum(isnan(testA_fix_nan)));
InfNum = sum(sum(isinf(testA_fix_nan)));
fprintf('nanNum = %d   InfNum = %d\n',nanNum, InfNum);
pause;
%===============Dimensionality reduction========
fprintf('==========Dimensionality reduction=========\n');
testAprocessNum1 = testA_fix_nan(:, 2:231);%232
testAprocessNum2 = testA_fix_nan(:, 233:751);%752
testAprocessNum3 = testA_fix_nan(:, 753:773);%774
testAprocessNum4 = testA_fix_nan(:, 775:944);%945
testAprocessNum5 = testA_fix_nan(:, 946:1701);%1702
testAprocessNum6 = testA_fix_nan(:, 1703:2381);%2382
testAprocessNum7 = testA_fix_nan(:, 2383:3693);%3694
testAprocessNum8 = testA_fix_nan(:, 3695:3893);%3894
testAprocessNum9 = testA_fix_nan(:, 3895:4168);%4169
testAprocessNum10 = testA_fix_nan(:, 4170:5977);%5978
testAprocessNum11 = testA_fix_nan(:, 5979:6191);%6192
testAprocessNum12 = testA_fix_nan(:, 6193:6574);%6575
testAprocessNum13 = testA_fix_nan(:, 6576:8027);

[testAnormProcessNum1, testAmu1, testAsigma1] = featureNormalize(testAprocessNum1);
[testAnormProcessNum2, testAmu2, testAsigma2] = featureNormalize(testAprocessNum2);
[testAnormProcessNum3, testAmu3, testAsigma3] = featureNormalize(testAprocessNum3);
[testAnormProcessNum4, testAmu4, testAsigma4] = featureNormalize(testAprocessNum4);
[testAnormProcessNum5, testAmu5, testAsigma5] = featureNormalize(testAprocessNum5);
[testAnormProcessNum6, testAmu6, testAsigma6] = featureNormalize(testAprocessNum6);
[testAnormProcessNum7, testAmu7, testAsigma7] = featureNormalize(testAprocessNum7);
[testAnormProcessNum8, testAmu8, testAsigma8] = featureNormalize(testAprocessNum8);
[testAnormProcessNum9, testAmu9, testAsigma9] = featureNormalize(testAprocessNum9);
[testAnormProcessNum10, testAmu10, testAsigma10] = featureNormalize(testAprocessNum10);
[testAnormProcessNum11, testAmu11, testAsigma11] = featureNormalize(testAprocessNum11);
[testAnormProcessNum12, testAmu12, testAsigma12] = featureNormalize(testAprocessNum12);
[testAnormProcessNum13, testAmu13, testAsigma13] = featureNormalize(testAprocessNum13);

[testAU1, testAS1, testAK1] = pca(testAnormProcessNum1);
[testAU2, testAS2, testAK2] = pca(testAnormProcessNum2);
[testAU3, testAS3, testAK3] = pca(testAnormProcessNum3);
[testAU4, testAS4, testAK4] = pca(testAnormProcessNum4);
[testAU5, testAS3, testAK5] = pca(testAnormProcessNum5);
[testAU6, testAS6, testAK6] = pca(testAnormProcessNum6);
[testAU7, testAS7, testAK7] = pca(testAnormProcessNum7);
[testAU8, testAS8, testAK8] = pca(testAnormProcessNum8);
[testAU9, testAS9, testAK9] = pca(testAnormProcessNum9);
[testAU10, testAS10, testAK10] = pca(testAnormProcessNum10);
[testAU11, testAS11, testAK11] = pca(testAnormProcessNum11);
[testAU12, testAS12, testAK12] = pca(testAnormProcessNum12);
[testAU13, testAS13, testAK13] = pca(testAnormProcessNum13);

testAreducedProcess1 = projectData(testAnormProcessNum1, testAU1, testAK1);
testAreducedProcess2 = projectData(testAnormProcessNum2, testAU2, testAK2);
testAreducedProcess3 = projectData(testAnormProcessNum3, testAU3, testAK3);
testAreducedProcess4 = projectData(testAnormProcessNum4, testAU4, testAK4);
testAreducedProcess5 = projectData(testAnormProcessNum5, testAU5, testAK5);
testAreducedProcess6 = projectData(testAnormProcessNum6, testAU6, testAK6);
testAreducedProcess7 = projectData(testAnormProcessNum7, testAU7, testAK7);
testAreducedProcess8 = projectData(testAnormProcessNum8, testAU8, testAK8);
testAreducedProcess9 = projectData(testAnormProcessNum9, testAU9, testAK9);
testAreducedProcess10 = projectData(testAnormProcessNum10, testAU10, testAK10);
testAreducedProcess11 = projectData(testAnormProcessNum11, testAU11, testAK11);
testAreducedProcess12 = projectData(testAnormProcessNum12, testAU12, testAK12);
testAreducedProcess13 = projectData(testAnormProcessNum13, testAU13, testAK13);

sumK = (testAK1 + testAK2 + testAK3 + testAK4 + testAK5 + testAK6 + testAK7 + testAK8 + testAK9 + testAK10 + testAK11 + testAK12 + testAK13)


fprintf('=============== training neural network =============\n');
pause;
testAX = [testA_fix_nan(:, 1), testAreducedProcess1, testA_fix_nan(:, 232), testAreducedProcess2, testA_fix_nan(:, 752), testAreducedProcess3, testA_fix_nan(:, 774), testAreducedProcess4, testA_fix_nan(:, 945), testAreducedProcess5, testA_fix_nan(:, 1702), testAreducedProcess6, testA_fix_nan(:, 2382), testAreducedProcess7, testA_fix_nan(:, 3694), testAreducedProcess8, testA_fix_nan(:, 3894), testAreducedProcess9, testA_fix_nan(:, 4169), testAreducedProcess10, testA_fix_nan(:, 5978), testAreducedProcess11, testA_fix_nan(:, 6192), testAreducedProcess12, testA_fix_nan(:, 6575), testAreducedProcess13];

