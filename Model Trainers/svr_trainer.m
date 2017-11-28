function [trained_model] = svr_trainer(X,Y,HP,Kernel)
%% Construct a surrogate model using Kriging (Using DACE toolbox)
    % HP(1) or SVR_CC is recommended to be between 0.1 and 100
    % HP(2) or SVR_GG is recommended to be between 0.1 and 10
    % Kernel can take on 4 values:
    %   'Linear', 'Polynomial', 'RBF', 'Sigmoid'
    
switch nargin
    case 2
        SVR_CC = 1.0; % default value of kernel parameter
        SVR_GG = 1.0; % default value of shape parameter
        Kernel = 'RBF'; % default kernel type
    case 3
        if length(HP) == 1,
            Kernel = 'Linear'; % only SVR kernel that uses 1 HP
            SVR_CC = HP(1); % default value of kernel parameter
        elseif length(HP) == 2,
            Kernel = 'RBF'; % default kernel type
            SVR_CC = HP(1); % default value of kernel parameter
            SVR_GG = HP(2); % default value of shape parameter
        end
    case 4
        if length(HP) == 1,
            SVR_CC = HP(1); % default value of kernel parameter
        elseif length(HP) == 2,
            SVR_CC = HP(1); % default value of kernel parameter
            SVR_GG = HP(2); % default value of shape parameter
        end
end

    switch(Kernel)
          
        case 'Linear'     
          model = svmtrain(Y,X,['-s 4 -t 0 -c' num2str(SVR_CC) '-n 0.5']);
        case 'Polynomial'
          model = svmtrain(Y,X,['-s 4 -t 1 -d 3 -r 0 -g ' num2str(SVR_GG) '-c' num2str(SVR_CC) '-n 0.5']);
        case 'RBF'
          model = svmtrain(Y,X,['-s 4 -t 2 -g ' num2str(SVR_GG) '-c' num2str(SVR_CC) '-n 0.5']);
        case 'Sigmoid' 
          model = svmtrain(Y,X,['-s 4 -t 3 -r 0 -g ' num2str(SVR_GG) '-c' num2str(SVR_CC) '-n 0.5']);
    end

    trained_model = @(X) svmpredict(1,X,model,'-q');
end

