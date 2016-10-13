


function    [Predict_label,dec,accuracy,auc,w] = ccSVM(X,train,test,y,L,lambda,C)

% 
% ccSVM - Support Vector Machine with confounder correction
%
%    [Predict_label,dec,accuracy,auc,w] = ccSVM(X,train,test,y,L,lambda,C)
%
%Input: 
% X: n by m matrix, m is the number of samples, n is the number of features
% y: the class label for all the data
% L: the kernel of all data only based on the side information
% train: the indices of training samples 
% test: the indices of test samples.
%Output:
% Predict_label:  The predicted labels for the test samples
% dec: the decision value for the test samples
% accuracy: the prediction accuracy
% auc: the area under curve of the prediction
% w: the weight vector based on training data


% File        : ccSVM.m
%
% Date        : 27th March 2011
%
% Author      : Limin Li
%


[X_new,K_new,l] = Rescaling(X,L,lambda);


model_new = svmtrain(y(train), [(1:length(train))' K_new(train,train)], ['-c ' num2str(C) ' -t 4']);
[Predict_label, accuracy, dec] = svmpredict(y(test), [(1:length(test))' K_new(test,train)], model_new); 
label = model_new.Label;
auc = ComputAuc(dec,y(test),label(1),label(2));


svind = model_new.SVs;
sv = X_new(:,svind);
alpha = model_new.sv_coef;
w_new = sv*alpha;
w = w_new./l;



end

