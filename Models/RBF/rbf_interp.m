function rbf_coeff = rbf_interp(x_in,y_out,C,KernelType)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to construct the RBF Interpolation surrogate model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('Training RBF: %s\n',KernelType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch nargin
    case 2
        C = 0.9; % default value of shape factor
        KernelType = 'Multiquadric'; % default kernel type
    case 3
        KernelType = 'Multiquadric'; % default kernel type
end

x = x_in';
mu = y_out;

n_p = size(x,2);          % number of data points
n_dim = size(x,1);        % number of dimensions

% Create radial basis function approximation
% n_p
a = zeros(n_p,n_p);
for i = 1:n_p
    for j = 1:n_p
        a(i,j) = radbas(x(:,i),x(:,j),C,KernelType);  
    end
end

rbf_coeff = a\mu;


