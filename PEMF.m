function [PEMF_Error, model] = PEMF( surrogate_trainer,X,Y, error_type, verbosity, n_pnts_final, n_pnts_step, n_steps, n_permutations)
%  version 2016.v1
%  Predictive Estimation of Model Fidelity (PEMF) is a model-independent 
%  approach to quantify surrogate model error.  PEMF takes as input a
%  model trainer, sample data on which to train the model, and hyper-
%  parameter values to apply to the model.  As output, it provides an 
%  estimate of the median or maximum error in the surrogate model.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    First time use instructions
%      >In order to call PEMF from anywhere, add path to the PEMF directory
%
%      >In order to use any 3rd party surrogate modeling package 
%       (e.g., "dace" for Kriging, or "Libsvm" for SVR),
%       put it inside the "/Models" subfolder
%
%      >Try the example demo_PEMF (under the Examples directory) to learn 
%	    how to use PEMF
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Required Inputs:
% 
%        X - x data, with each row of x corresponding to a single data point
% 
%        Y - y data, a single column where each element corresponds to a row 
%            of x.
%  
%        surrogate_trainer - 
%            This input is a function which trains and produces a callable
% 	         surrogate model.  It takes as input the data for the 
% 	         model and produces as output a function handle which will
% 	         give the model's prediction of Y for a given X. The 
% 	         trainer function should be of the following form:
%           
%              function [trained_model] = train_model(X_data,Y_data)
% 			   coeffs = ... fit a model...
% 			   trained_model = @(x) call_to_model(coeffs,x)
% 	           end
%                
%            The input to train_model is a set of data, X and Y. The 
%            output of train_model is the function, trained_model, 
%            trained_model(x1) should return an estimate of y at x1. 
%            Subsets of the X and Y data given to PEMF will 
%            be used to train models.  This is why PEMF needs the 
%            trainer and not just the model.  Matlab's fit function 
% 	         is compatible as a trainer.  If you're confused, experiment
% 	         with "fit" first.  Note that it produces a callable 
% 	         function as an output. Example code for "fit" on 2D data:
%           
%              X = 1:.1:10; Y = log(X);
%              surrogate_trainer = @(X,Y) fit(X,Y,'smoothingspline');
%              err = PEMF(surrogate_trainer, X,Y);
% 
% 	         Note that PEMF does not want a trained model, but the 
% 	         model trainer itself.  The first input to PEMF, surrogate_
% 	         trainer, should be @train_model (following the convention
% 	         shown above).
% 
%            Any model hyper-parameters or kernel settings should be 
%            handled by using anonymous functions (default values are used 
%            if hyperparameter/kernel choices are not specified)
%            An example, of calling RBF with the multiqudric kernel and a
%            shape factor of 0.9, is shown below:
%                
%              surrogate_trainer = ...
%                @(X,Y) train_model(X, Y, 0.9, 'Multiquadric')
%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Optional Inputs:
%        error_type  - type of error. Options:
%            'median' - modal value of median error
%            'max'    - modal value of maximum error  
%            'both'   - modal values of both errors
%        verbosity   - This input tells PEMF how much output you want.  
% 		     'high' - plot & command line output 
%            'low' - command line output
%            'none' - no command line or plot output 
%        n_pnts_final - number of test points in the last iteration
%        n_pnts_step - step size in predictive error model
%        n_steps - number of steps in predictive error model 
%        n_permutations - number of combinations tried in each step
% 
%    Default behavior of optional inputs:
%        error_type - median
%        verbosity - default
%        n_pnts_final - max(5% of sample points, 3)
%        n_pnts_step - equal to n_pnts_final
%        n_steps - 5
%        n_permutations - 40 (recommended: between 10 to 40)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Outut: 
%        PEMF_Error - Estimation of model error
%        model - Final surrogate model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    For further explaination of PEMF's optional inputs, refer to the 
%       PEMF paper. "Predictive quantification of surrogate model fidelity
%       based on modal variations with sample density." by Chowdhury and 
%       Mehmani. DOI: 10.1007/s00158-015-1234-z
%    The above article and its citation (bibtex) can be found at:
%       http://adams.eng.buffalo.edu/publications/
%    
%    Cite PEMF as:
%       A. Mehmani, S. Chowdhury, and A. Messac, "Predictive quantification
%       of surrogate model fidelity based on modal variations with sample 
%       density," Structural and Multidisciplinary Optimization, vol. 52, 
%       pp. 353-373, 2015.
%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Changelog (Newest at top)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date 20-Nov-2016
% Modified by: Ben Rinauto
% brinauto@buffalo.edu
% - Removed unused variables
% - Renamed variables for clarity
% - Added optional plotting
% - Added commandline output
% - Added optional variables and default behaviors
% - Modified PEMF to take any surrogate model
% - Added input checking
% - Added better plotting
% - Moved log-normal fitting to subfunction
% - Changed separate .m files to subfunctions here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 4-Apr-2016
% Modified by: Ben Rinauto
% brinauto@buffalo.edu
% - Improved speed with preallocation during for loops
% - Improved speed by taking X and Y as arguments instead of loading
%   from hard disc
% - Improved speed by adding option for calculating one or both objectives
% - Improved speed by commenting out unused statistical variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add Paths to all subfolders (i.e. MODELS)
here = mfilename('fullpath');
[path, ~, ~] = fileparts(here);
addpath(genpath(path));

%% Set default variable behavior 
% (inelegant solution, but I don't know how to use matlab's inputchecker)
n_var = size(X,2);  % number of dimensions
n_pnts = size(X,1);  % number of points

if(nargin < 3)
    error('Too few input arguments');
end

% A lot of input checks to go along with setting defaults
switch nargin
    case 3
        error_type = 'median';
        verbosity = 'low';
        n_pnts_final = floor(max(0.05*n_pnts,3));
        n_pnts_step = n_pnts_final;
        n_steps = 4;
        n_permutations = 40;
    case 4
        if(isempty(error_type)), error_type = 'median'; end
        verbosity = 'low';
        n_pnts_final = floor(max(0.05*n_pnts,3));
        n_pnts_step = n_pnts_final;
        n_steps = 4;
        n_permutations = 40;
    case 5
        if(isempty(error_type)), error_type = 'median'; end
        if(isempty(verbosity)), verbosity = 'none'; end
        n_pnts_final = floor(max(0.05*n_pnts,3));
        n_pnts_step = n_pnts_final;
        n_steps = 4;
        n_permutations = 40;
    case 6        
        if(isempty(error_type)), error_type = 'median'; end
        if(isempty(verbosity)), verbosity = 'none'; end
        if(isempty(n_pnts_final)), n_pnts_final = floor(max(0.05*n_pnts,3)); end
        n_pnts_step = n_pnts_final;
        n_steps = 4;
        n_permutations = 40;
    case 7
        if(isempty(error_type)), error_type = 'median'; end
        if(isempty(verbosity)), verbosity = 'none'; end
        if(isempty(n_pnts_final)), n_pnts_final = floor(max(0.05*n_pnts,3)); end
        if(isempty(n_pnts_step)), n_pnts_step = n_pnts_final; end
        n_steps = 4;
        n_permutations = 40;
    case 8
        if(isempty(error_type)), error_type = 'median'; end
        if(isempty(verbosity)), verbosity = 'none'; end
        if(isempty(n_pnts_final)), n_pnts_final = floor(max(0.05*n_pnts,3)); end
        if(isempty(n_pnts_step)), n_pnts_step = n_pnts_final; end        
        if(isempty(n_steps)), n_steps = 4; end
        n_permutations = 40;
    case 9
        if(isempty(error_type)), error_type = 'median'; end
        if(isempty(verbosity)), verbosity = 'none'; end
        if(isempty(n_pnts_final)), n_pnts_final = floor(max(0.05*n_pnts,3)); end
        if(isempty(n_pnts_step)), n_pnts_step = n_pnts_final; end        
        if(isempty(n_steps)), n_steps = 4; end
        if(isempty(n_permutations)), n_permutations = 40; end
end

% More input checks
model = check_input(surrogate_trainer, X,Y,error_type, verbosity, ...
    n_pnts_final, n_pnts_step, n_steps, n_permutations);

%% PEMF
% Display PEMF starting
if(~strcmp(verbosity,'none'))
    disp('PEMF Starting');
end

% Define lower bounds (LB) & upper bounds (UB) on data
LB = zeros(n_var,1);
UB = zeros(n_var,1);
for j=1:n_var
    LB(j)=min(X(:,j));
    UB(j)=max(X(:,j));
end

y_ref = std(Y); % Used as divisor for relative error measurement
data=[X,Y];

PEMF_Error_max = zeros(1,n_steps); % Preallocate
PEMF_Error_med = zeros(1,n_steps); % Preallocate
n_train = zeros(1,n_steps); % Preallocate
MedianTest = zeros(n_steps, n_permutations); %Preallocate
MaxTest = zeros(n_steps, n_permutations); %Preallocate
med_params = zeros(n_steps,2);
max_params = zeros(n_steps,2);
for i=1:n_steps
    n_train(i) = n_pnts - (n_pnts_final+(i-1)*n_pnts_step);
    
    % Training and Test Points for all combinatio in i-th step  
    M_Combination = zeros(n_permutations,n_train(i)); % Preallocate for speed
    for i_NC = 1:n_permutations
        M_Combination(i_NC,:)=randsample(n_pnts,n_train(i));
    end
    
    % Parallel support
    
    for j=1:n_permutations
        % Separate training points and test points:
        training_data = data(M_Combination(j,:),:);
        test_data = data;
        test_data(M_Combination(j,:),:)=[]; % remove the rows in the training data
        
        % Define Training and Test Points (X and Y)
        x_train = training_data(:,1:n_var);
        y_train = training_data(:,n_var+1);
        n_tests = size(test_data,1);
        x_test = test_data(:,1:n_var);
        y_test = test_data(:,n_var+1);
        
        % Train Model and Test it
        if(strcmp(verbosity,'high'))
            trained_model = surrogate_trainer(x_train,y_train);
        else
            [~,trained_model] = evalc('surrogate_trainer(x_train,y_train)');
        end
        
        RAE = zeros(1,n_tests); % RAE - Relative Absolute Error
        for k = 1:n_tests
            y_predicted = trained_model(x_test(k,:));
             RAE(k) = abs((y_test(k)-y_predicted)/y_ref);
        end        
        
        % Calculate Median/Max of RAE
        MedianTest(i,j)=   median(RAE);
        MaxTest(i,j)   =   max(RAE); 
    
    end % end for m combinations

    if(strcmp(error_type,'median') || strcmp(error_type,'both'))
        % MODE - MED
        % Remove Outlier in Med(RAE) and fit to log-normal
        parmhat = lognfit_outliers(MedianTest(i,:),70); 
        med_params(i,:) = parmhat;
        % Calculate mode of distribution
        PEMF_Error_med(i)=exp(parmhat(1)-(parmhat(2))^2);
    
    end % end if median

    if(strcmp(error_type,'max') || strcmp(error_type,'both'))
        % MOD-Max
        % Remove Outlier in Max(RAE) and fit to log-normal
        parmhat = lognfit_outliers(MaxTest(i,:),60); 
        max_params(i,:) = parmhat;
        % Mode of Max Estimation
        PEMF_Error_max(i)=exp(parmhat(1)-(parmhat(2))^2);
        
    end % end if max
    if(strcmp(verbosity,'low') || strcmp(verbosity,'high'))
        tot = n_steps*n_permutations;
        curr = i*n_permutations;
        fprintf('Iter %d: %d of %d intermediate models evaluated\n', ...
            i, curr, tot);
    end
end % end for n steps

n_train=flipud(n_train(:));
PEMF_Error_med=flipud(PEMF_Error_med(:));
PEMF_Error_max=flipud(PEMF_Error_max(:));
MaxTest = flipud(MaxTest);
MedianTest = flipud(MedianTest);
max_params = flipud(max_params);
med_params = flipud(med_params);

%% Select the best fitting model for error and predict the total error
for model_type = 1:2
    [RMe,~,~]=SelectRegression(n_train,PEMF_Error_med,model_type,n_pnts);
    RMSE_MedianE(model_type,:)=RMe(:);
    [RMa,~,~]=SelectRegression(n_train,PEMF_Error_max,model_type,n_pnts);
    RMSE_MaxE(model_type,:)=RMa(:);
end

%%  Median Error
[~,model_id] = min(RMSE_MedianE);
model_type_med = model_id;
[~,MedianPrediction,v_med] = SelectRegression(n_train,PEMF_Error_med,model_type_med,n_pnts);
CorrelationParameterMedian = SmoothnessCriteria(n_train,PEMF_Error_med,model_type_med);
if abs(CorrelationParameterMedian)>=0.90
    PEMF_MedError_return = MedianPrediction;
    x_med=[n_train;n_pnts];
else
    PEMF_MedError_return = PEMF_Error_med(n_steps);
    x_med=[n_train;n_train(end)];
    if(strcmp(error_type,'both') || strcmp(error_type,'median'))
        fprintf('\nSmoothness criterion violated for predicition of median error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end

%%  Maximum Error
[~,model_id] = min(RMSE_MaxE);
model_type_max = model_id;
[~,MaxPrediction,v_max] = SelectRegression(n_train,PEMF_Error_max,model_type_max,n_pnts);
CorrelationParameterMax = SmoothnessCriteria(n_train,PEMF_Error_max,model_type_max);
if abs(CorrelationParameterMax)>=0.90
    PEMF_MaxError_return = MaxPrediction;
    x_max=[n_train;n_pnts];
else
    PEMF_MaxError_return = PEMF_Error_max(n_steps);
    x_max=[n_train;n_train(end)];
    if(strcmp(error_type,'both') || strcmp(error_type,'max'))
        fprintf('\nSmoothness criterion violated for predicition of maximum error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end    


% Plot
PEMF_Error_med=[PEMF_Error_med;PEMF_MedError_return];
PEMF_Error_max=[PEMF_Error_max;PEMF_MaxError_return];
if(strcmp(verbosity,'high'))
    if(strcmp(error_type,'both') || strcmp(error_type,'median'))
        fig = plot_pemf(x_med,PEMF_Error_med,med_params, model_type_med, v_med);
        set(get(get(fig,'CurrentAxes'),'Title'),'String','Median Relative Absolute Error')
    end
    if(strcmp(error_type,'both') || strcmp(error_type,'max'))
        fig = plot_pemf(x_max,PEMF_Error_max,max_params, model_type_max, v_max);
        set(get(get(fig,'CurrentAxes'),'Title'),'String','Maximum Relative Absolute Error')
    end
end

%% Return desired type of error
if(strcmp(error_type,'median'))
    PEMF_Error = PEMF_MedError_return;
    if(~strcmp(verbosity,'none'))
        fprintf('\nPEMF_Error (median): %f\n\n',PEMF_Error)
    end
elseif(strcmp(error_type,'max'))
    PEMF_Error = PEMF_MaxError_return;
    if(~strcmp(verbosity,'none'))
        fprintf('\nPEMF_Error (max): %f\n\n',PEMF_Error)
    end
elseif(strcmp(error_type, 'both'))
    PEMF_Error = [PEMF_MedError_return,PEMF_MaxError_return];
    if(~strcmp(verbosity,'none'))
        fprintf('\nPEMF_Error (median): %f\n\n',PEMF_Error(1))
        fprintf('\nPEMF_Error (max): %f\n\n',PEMF_Error(2))
    end
end



end

%%%%%%%%%%%%%%%%%% Sub Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function parmhat = lognfit_outliers(dat, outlier_percent)
        YP = prctile(dat,outlier_percent);
        len=1;
        CT = zeros(length(dat));
        for k=1:max(length(dat))
            if dat(k) <= YP
                CT(len)= dat(k);
                len=len+1;
            end
        end    
        CT(len:end) = [];

        CP = CT';
        mu = mean (CP);
        sigma = std (CP);
        [n,~] = size(CP);
        Meanmat = repmat (mu , n , 1);
        Sigmamat = repmat (sigma, n, 1);
        outliers = abs (CP - Meanmat) > 3* Sigmamat ;
        CP( any( outliers, 2), :) = [];     
        parmhat = lognfit(CP(:)); 
end

function fig = plot_pemf(X,Y, lognfit_params, reg_type, reg_fit_param)
    
    fig = figure;
        hold on;
        calc = plot(X(1:end-1),Y(1:end-1),'bo','LineWidth',2); 
        pred = plot(X(end),Y(end),'kx','LineWidth',2);  
        
        mu1 = lognfit_params(1,1);
        sig1 = lognfit_params(1,2);
        mode_x1 = exp(mu1-sig1^2);
        
        % create, rotate, and scale distributions
        npnts = 100; step = X(2)-X(1);
        for i=1:1:length(lognfit_params)
            mu = lognfit_params(i,1);
            sig = lognfit_params(i,2);
            mode_x = exp(mu-sig^2);
            mode_p = lognpdf(mode_x,mu,sig);
            plot_tune = 0.05;
            xmax = fzero(@(x) lognpdf(x,mu,sig)-plot_tune*mode_p,[mode_x,1000*mode_x]);
            xmax = min(xmax,3*mode_x1);
            xs = 0:xmax/(npnts-1):xmax;
            for j = 1:1:length(xs)
                ys(j) = lognpdf(xs(j),mu,sig);
            end
            step_frac = 0.75;
            ys = ys.*(step_frac*step/max(ys));
            pnts = [xs',ys'];
            R = [cosd(90),sind(90);-sind(90),cosd(90)]; % rotates 90 deg
            pnts = pnts*R; % rotates 90 deg
            pnts(:,1) = pnts(:,1)+X(i);
            dist = plot(pnts(:,1),pnts(:,2),'k:');
            plot([pnts(1,1),pnts(1,1)],[pnts(2,2),pnts(end,2)],'k:');
        end
        
        xs = X(1):0.01:X(end);
        ys = 0*xs;
        a = reg_fit_param(1); b = reg_fit_param(2);
        if(reg_type == 1)
            ys = a.*exp(b.*xs);
        elseif(reg_type == 2)
            ys = a.*(xs).^(b);
        end
        
        fitted = plot(xs,ys,'--','LineWidth',2);
        
        
        hold off;
        legend([dist,calc,pred],'Error Distribution','Estimated Intermediate Error','Predicted Final Error');
        title('PEMF Error'); ylabel('Relative Absolute Error');
        if(Y(end) == Y(end-1))
            n_points = X(end) + step;
        else
            n_points = X(end);
        end
        s = strcat({'Number of training points (out of '},num2str(n_points),{' total data points)'});
        xlabel(s{1});
     
end

function [model] = check_input(surrogate_trainer, X,Y, error_type, verbosity, n_pnts_final, n_pnts_step, n_steps, n_permutations)
% Check X and Y data are the right size
if(size(Y,1) ~= size(X,1))
    error('X and Y must have the same number of columns');
end

% Check is there enough data
min_pnts_first_step = 3;
if(length(Y) < n_steps*n_pnts_step + min_pnts_first_step)
    error('Not enough data points.  Use less steps, a smaller step size, or provide more data');
end

% Check n_permutations is greater than 9
if(n_permutations < 10 )
    error('At least 10 permutations are required per iteration')
end

% Check trainer
if(~isa(surrogate_trainer,'function_handle'))
    error('surrogate_trainer must be a function handle');
end

% Test trainer to make sure it returns a surrogate model of the correct
% form.  First make sure that surrogate_trainer can be called.
try
    model = surrogate_trainer(X,Y);
catch ME
    msg = 'PEMF surrogate_trainer does not match expected format';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% Check that output of surrogate_trainer is of the form y = model(X).
try
    model(X(1,:));
catch ME
    msg = 'PEMF trained_model (output of surrogate_trainer) does not match expected format';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% Check error is 'median', 'max', or 'both'
if(~(strcmp(error_type,'median') || strcmp(error_type,'max') || strcmp(error_type,'both')))
    error('error_type must be ''median'', ''max'', or ''both''');
end

% Check verbosity is 'high', 'low', or 'none'
if(~(strcmp(verbosity,'high') || strcmp(verbosity,'low') || strcmp(verbosity,'none')))
    error('verbosity must be ''high'', ''low'', or ''none''');
end

% Check Y data is a single column vector
if(size(Y,2) ~= 1)
    error('Y data should be a single column vector.');
end

% Warn if data spans ~3 orders of magnitude
if( min(Y)/max(Y) < 5*10^-3)
    warning('Data spans much more than 2 orders of magnitude.  PEMF uses relative error.');
end

end

function Rho = SmoothnessCriteria(x,y,iSType)
% Returns the smothness of the fit for a given regression model
if iSType==1
    [R,~] = corrcoef(x,log(y));
end

if iSType==2
    [R,~] = corrcoef(log(x),log(y));
end

Rho = R(1,2);
end % Smoothness Criteria

function [RMSE,ErrorPrediction,VCoe] = SelectRegression(X,Y,iSType,NinP)
% Helper function with a few modes
% - Able to fit PEMF error to two different regression models depending on
%       the value of isType 
% - Able to predict the next value in the regression (ErrorPrediction)
% - Able to give the model fit parameters (VCoe = [a,b])
%% 1. Exponential Fit Model   Y=a*exp(b*X)
if iSType==1
    
    ff = fit(X,Y,'exp1');
    a11=ff.a;b11=ff.b;
    ErrorPrediction=a11*exp(b11*NinP);
    
    data=Y;
    for j=1:max(size(X))
       estimate(j)=a11*exp(b11*X(j)); 
    end
   
    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
%% 2. Power Fit Model   a*X^b

if iSType==2
     [a11,b11]=PowerFit(Y,X);
     ErrorPrediction=a11*(NinP)^(b11);
     
    data=Y;
    for j=1:max(size(X))
       estimate(j)=a11*X(j)^(b11); 
    end
     
    
%%% RMSE evaluation

    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
end % SelectRegression

function r = Grmse(data,estimate)
% Function to calculate root mean square error from a data vector or matrix 
% I = ~isnan(data) & ~isnan(estimate); 
% data = data(I); estimate = estimate(I);
rI=0;
for I=1:max(size(data))
    rI=rI+(data(I)-estimate(I)).^2;
end
RI=rI/(max(size(data)));
r=sqrt(RI);

end %Grmse

function [a,b] = PowerFit(Y,X)
% Performs a regression fit with a Power fit model Y = a*X^b

n=length(X);
Z=zeros(1,n);
for i=1:n
    Z(i)=log(Y(i));
end
w=zeros(1,n);
for i=1:n
    w(i)=log(X(i));
end
wav=sum(w)/n;
zav=sum(Z)/n;
sum(Z);
Swz=0;
Sww=0;
for i=1:n
    Swz=Swz +w(i)*Z(i)-wav*zav;
    Sww=Sww + (w(i))^2-wav^2;
end

a1=Swz/Sww;
a0=zav-a1*wav;
a=exp(a0);
b=a1;


xp=(0:0.001:max(X));
yp=zeros(1,length(xp));
for i=1:length(xp)
yp(i)=a.*(xp(i)^b);
end
end % Power Fit
