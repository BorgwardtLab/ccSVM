%ParameterSetting: to select parameters lambda and C by k-fold cross validation

% Input: 
% X: training data
% y: the class label for the training data
% L: the kernel for training data only based on the side information
% LambdaRange: the range of lambda
% CRange: the range of C
% kfold: k-fold crossvalidataion
% eg. LambdaRange = [1e+8,1e+4,1e+2,1,1e-2,1e-4,1e-8];
%     CRange = 2.^[-8,-4,-2,0,2,4,8];

% File        : ccSVM.m
%
% Date        : 27th March 2011
%
% Author      : Limin Li
%

function  [lambda,C] = ParameterSetting(X,y,L,LambdaRange,CRange,kfold)


[n,m] = size(X);
CVO = cvpartition(m,'k',kfold);


%% choose best C

K = X'*X;
for i = 1: length(CRange)

c = CRange(i);


for j = 1:kfold
    
    test = find(CVO.test(j));
    train = find(CVO.training(j)); 
    kcauc(i,j) = svm(y,train,test,K,c);

end


end

[a,b]= max(mean(kcauc,2));

C = CRange(b);

%%Choose best lambda

for i = 1: length(LambdaRange)

lam = LambdaRange(i);

[X_new,K_new,l] = Rescaling(X,L,lam);


for j = 1:kfold   
    test = find(CVO.test(j));
    train = find(CVO.training(j)); 
    kauc(i,j) = svm(y,train,test,K_new,C);

end


end

[a,b]= max(mean(kauc,2));
lambda = LambdaRange(b);


end