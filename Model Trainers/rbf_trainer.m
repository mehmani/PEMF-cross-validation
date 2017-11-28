function [trained_model] = rbf_trainer(X_data, Y_data, C, kernel_type)
% Trains a radial basis function returns the
% trained model
%
% kernel_type: 'linear', 'Cubic', 'Tps', 'Gaussian', or 'Multiquadric'
% C: shape factor (if 'Gaussian' or 'Multiquadric' kernel is used)
%    recommended values: 0.1 < C < 3.0
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch nargin
    case 2
        C = 0.9; % default value of shape factor
        kernel_type = 'Multiquadric'; % default kernel type
    case 3
        kernel_type = 'Multiquadric'; % default kernel type
end

rbf_coeff = rbf_interp(X_data,Y_data,C,kernel_type);

trained_model = @(X) rbf_approx(X_data, X, rbf_coeff,C,kernel_type);

end