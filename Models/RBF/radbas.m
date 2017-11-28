function d = radbas(x1,x2,C,KernelType)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to construct the RBF Interpolation surrogate model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Souma Chowdhury
% soumacho@buffalo.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch(KernelType)
    
    case 'Linear'
    d = norm(x1-x2);
    
    case 'Cubic'
    d = (norm(x1-x2))^3;
    
    case 'Tps'
    % Thin plate spline
    r=(norm(x1-x2));
        if r < 1e-200
        d=0;
        else   
        d = r^2*log(r);
        end
        
    case 'Gaussian'
    d = exp(-(norm(x1-x2))^2/(2*C));
    
    case 'Multiquadric'
    d = ((norm(x1-x2))^2+C^2)^0.5;
end