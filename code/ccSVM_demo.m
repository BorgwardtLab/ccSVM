% A Demo to show how to use ccSVM.
clear;


%add libsvm to matlab path
addpath('./libsvm-mat-3.0-1');

%load data with X,y and L.
load('DemoData.mat','X','y','L');

%%TO choose training and test dataset
[n,m] = size(X);
CVO = cvpartition(m,'k',5);
test = find(CVO.test(1));
train = find(CVO.training(1)); 


%the setting up to select parameter lambda and C
LambdaRange = [1e-8,1e-4,1e-2,1,1e+2,1e+4,1e+8];
CRange = 2.^[-8,-4,-2,0,2,4,8];
kfold = 2;

%parameter selection by kfold cross validation based only on training data
[lambda,C] = ParameterSetting(X(:,train),y(train),L(train,train),LambdaRange,CRange,kfold);


%to do prediction on test data using ccSVM
[Predict_label,dec,accuracy,ccauc,w] = ccSVM(X,train,test,y,L,lambda,C);


%to do prediction on test data using standard SVM, setting lambda as 0
[Predict_label,dec,accuracy,svmauc,w] = ccSVM(X,train,test,y,L,0,C);


%bye bye