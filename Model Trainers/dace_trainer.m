function [trained_model] = dace_trainer(X,Y,HP,Kernel)
%% Construct a surrogate model using Kriging (Using DACE toolbox)
    % HP is recommended to be between 0.1 and 20
    % Kernel can take on 5 values:
    %   'Gaussian', 'Exponential', 'Cubic', 'Linear', and 'Spherical'
    
switch nargin
    case 2
        HP = 1.0; % default value of shape factor
        Kernel = 'Gaussian'; % default kernel type
    case 3
        Kernel = 'Gaussian'; % default kernel type
end

    switch(Kernel)
        case 'Gaussian'
            [dmodel,~] = dacefit(X, Y, @regpoly0, @corrgauss, HP);
        case 'Exponential'
            [dmodel,~] = dacefit(X, Y, @regpoly0, @correxp, HP);
        case 'Cubic'
            [dmodel,~] = dacefit(X, Y, @regpoly0, @corrcubic, HP);
        case 'Linear'
            [dmodel,~] = dacefit(X, Y, @regpoly0, @corrlin, HP);
        case 'Spherical'
            [dmodel,~] = dacefit(X, Y, @regpoly0, @corrspherical, HP);    
    end

    trained_model = @(X) predictor(X,dmodel);
end