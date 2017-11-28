function [f] = branin(X, c1, c2, c3, c4, c5, c6)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Branin function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If c1, c2, ... c6 are not given, default values are used.
if nargin<=6,
    c6 = 1 / (8*pi);
end

if nargin<=5,
	c5 = 10;
end

if nargin<=4,
	c4 = 6;
end

if nargin<=3,
	c3 = 5/pi;
end

if nargin<=2,
        c2 = 5.1 / (4*pi^2);
end

if nargin==1,
        c1 = 1;
end

y1 = c1 * (X(2) - c2*X(1)^2 + c3*X(1) - c4)^2;
y2 = c5*(1-c6)*cos(X(1));

f = y1 + y2 + c5;

end
