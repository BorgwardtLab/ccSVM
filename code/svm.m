function auc = svm(y,train,test,kernel,c)
                    y_train = y(train);
                    y_test = y(test);
                    
                    model = svmtrain(y_train, [(1:length(y_train))' kernel(train,train)], ['-c ' num2str(c) ' -t 4']);
                    
                    label = model.Label;
                    kpos_label = label(1);
                    kneg_label = label(2);
                    
                    [predict_label, accuracy, dec] = svmpredict(y_test, [(1:length(test))' kernel(test,train)], model);  
                                           
        
                    auc = ComputAuc(dec,y_test,kpos_label,kneg_label);
                    
end

