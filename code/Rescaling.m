% To do rescaling on the original data        
function [X_new,K_new,l] = Rescaling(X,L,lambda)


% Rescaling - To do feature rescaling on training data
%
%    [X_new,K_new,l] = Rescaling(X,L,lambda)

% File        : ccSVM.m
%
% Date        : 27th March 2011
%
% Author      : Limin Li
%

[n,m] = size(X);

H = eye(m,m)-1/m*ones(m,m);
L = H*L*H/((m-1)^2);

if lambda > 0
    for i = 1: n
        xi = X(i,:);
        l(i) = sqrt(lambda*xi*L*xi'+1);
        X(i,:) = xi/l(i);
    end

    l = l';

else
    
    l = ones(n,1); 

end


X_new = X;

K_new = X_new'*X_new;

