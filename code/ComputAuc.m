
%input:
% f:
% class:
% positive label
% negative label:
function auc = ComputAuc(f,class,pos_label,neg_label)
    [f_sorted,I] = sort(-f);
    f_sorted = -f_sorted;
    class_sorted = class(I);
    NUM_pos = length(find(class_sorted==pos_label));
    NUM_neg = length(find(class_sorted==neg_label));
    S_Neg = sum(find(class_sorted==neg_label));
    auc = (S_Neg-NUM_neg*(NUM_neg+1)/2)/(NUM_pos*NUM_neg);
end
    