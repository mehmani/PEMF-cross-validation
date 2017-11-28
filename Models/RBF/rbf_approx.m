function f = rbf_approx(x_in,x_test,rbf_coeff,C,KernelType)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to construct the RBF Interpolation surrogate model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Souma Chowdhury
% soumacho@buffalo.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = rbf_coeff;

x = x_in';
x_t = x_test';

n_p = size(x,2);          % number of data points
n_dim = size(x,1);        % number of dimensions

p = size(x_t,2);
k = 0;
%rms_sum = 0;

% Evaluate the test data
for i = 1:p        
        sum1 = 0; 
        
        for h = 1:n_p                      
            sum1 = sum1 + alpha(h)*radbas(x_t(:,i),x(:,h),C,KernelType); 
        end  
        
        f(i) = sum1;
end
     
