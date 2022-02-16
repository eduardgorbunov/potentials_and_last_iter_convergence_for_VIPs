clc; clear all;
verbose = 1;
%
%   min_x f(x)   with f: L-smooth and convex
%
%
%   x_{k+1} = x_k - gamma_k g_k   with g_k = \nabla f(x_k)
%
%
%   we study worst-case behavior of f(x_N) - f_*
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x0 g0 g1 ... gN ]   and G = P^T P
%   - F = [    f0 f1 ... fN ]
%
%   Define a few things for simplifying the code:
%   - \bar{x}_k :  x_k = P \bar{x}_k   (with k\in\{*,0,...N\})
%   - \bar{g}_k :  g_k = P \bar{g}_k
%  why this choice?
%  first: <g_i;x_k> = \bar{g}_i^T P^T P \bar{x}_k
%                   = \bar{g}_i^T G \bar{x}_k
%                   = Trace (G \bar{x}_k \bar{g}_i^T )
%  second:
%       \bar{x}_{k+1} = \bar{x}_k - gamma_k \bar{g}_k
%
%   - \bar{f}_k :  f_k = F \bar{f}_k

% algorithm setup:
N = 4;
L = 1;
R = 1; % this is the bound on ||x_0-x_*||^2

gamma = @(k)(1/L);

% internal notation:

dimG    = N + 2;        % dimension of G
dimF    = N + 1;        %
nbPts   = N + 2; % this is x_*, x_0, ..., x_N

barxstar = zeros(dimG,1);
bargstar = zeros(dimG,1);
barfstar = zeros(dimF,1);

barx0 = zeros(dimG, 1); barx0(1) = 1;

bargk = zeros(dimG, N+1); bargk(2:end,:) = eye(N+1); % so that bargk(:,i) returns gradient of x_{i-1}
barxk = zeros(dimG, N+1); barxk(:,1) = barx0;

ykminus1 = barx0; % this is for accelarated version
for i = 1:N
    barxk(:,i+1) = barxk(:,i) - gamma(i) * bargk(:,i);
    %     yk           = barxk(:,i) - gamma * bargk(:,i);
    %     barxk(:,i+1) = yk + (i-1)/(i+2) * (yk-ykminus1);
    %     ykminus1 = yk;
end
barfk = eye(N+1); % so that barfk(:,i) returns function value of x_{i-1}


barx = [barxstar barxk];
barg = [bargstar bargk];
barf = [barfstar barfk];

%% SDP

% this uses YALMIP:
G = sdpvar(dimG); % this defines a symmetric matrix variable, called G
F = sdpvar(1,dimF); % this defines the vector variable

constraint = ( G >= 0); % G is PSD
constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

objective = F * (barfk(:,end) - barfstar); % this is f(x_N) - f_*

% this is for the constraints on f
for i = 1:nbPts
    for j = 1:nbPts
        if i~=j
            constraint = constraint + ( F*(barf(:,j)-barf(:,i)) + barg(:,j)'*G*(barx(:,i)-barx(:,j)) + 1/2/L * (barg(:,i)-barg(:,j))'*G*(barg(:,i)-barg(:,j)) <= 0);
            %  0 >= f_j - f_i + g_j^T (x_i-x_j) + 1/2/L * ||g_i-g_j||^2
        end
    end
end
%

options = sdpsettings('verbose',verbose,'solver','MOSEK');
optimize(constraint,-objective,options);

[L*R^2/(4*N+2) double(objective)]
















