%
% This code aims at verifying (numerically) the first inequality from
% Theorem 1 of the paper
%   "List-Iterate Convergence of Optimistic Gradient Method for Monotone
%       Variationnal Inequalities".
%
% PROBLEM SETUP:
% Consider the problem of finding a zero of a monotone Lipschitz operator:
%       find x such that F(x) = 0
% where F is monotone and L-Lipschitz. We denote by x^* a solution to this
% problem.
%
% ALGORITHM: 
% The past extragradient method is described by two sets of iterates:
% x^k, tx^k where k denotes the iteration counter, as follows:
% initialize tx^0 = x^0 and x^1  = x^0 - gamma * F(x^0), then run
% for k=1,...,N-1:
%   tx^k    = x^k - gamma * F(tx^{k-1})
%   x^{k+1} = x^k - gamma * F(tx^k)
%
% Denotes p_k := ||F(x^{k})||^2 + 2 || F(x^{k}) - F(tx^{k-1}) ||^2 ;
% the code compute the maximum (i.e., worst-case) value of
%   ||x^{k+1} - x^* ||^2 + A_{k+1} * gamma^2 * p_{k+1}
%                       - ||x^{k}-x^*||^2 - A_k * gamma^2 * p_k,
% when A_{k+1} = A_k + 1/3. The expression should always be <= 0 for
% verifying the identity from Theorem 1 (with gamma<=1/3/L and A_k>=32/3).
% In the code below, we use k = 1 for notational convenience.

clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters: MODIFY HERE!

% pick the parameters for which you want to verify
% the inequality (numerically)
L = 1;
gamma = 1/3/L; 
A1 = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

verbose = 2;
tolerance = 1e-8;

% (0) Initialize an empty PEP
P=pep();

% (1) Set up the problem class
paramF.L  =  L; paramF.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
F = P.DeclareFunction('LipschitzStronglyMonotone',paramF);

xs  = F.OptimalPoint();  % this is some x^*

% (2) Set up the starting points
tx0 = P.StartingPoint(); % this is tx^0
x1  = P.StartingPoint(); % this is x^1

Ftx0    = F.gradient(tx0);
Fx1     = F.gradient(x1);
tx1     = x1 - gamma * Ftx0; 
Ftx1    = F.gradient(tx1);
x2      = x1 - gamma * Ftx1;
Fx2     = F.gradient(x2);

p1 = Fx1^2 + 2 * ( Fx1 - Ftx0 )^2;
p2 = Fx2^2 + 2 * ( Fx2 - Ftx1 )^2;
A2 = A1 + 1/3;
expression1 = ( x1-xs )^2 + A1 * gamma^2 * p1;
expression2 = ( x2-xs )^2 + A2 * gamma^2 * p2;
% (4) Set up the performance measure:
expression_to_verify = expression2 - expression1;
P.PerformanceMetric(expression_to_verify);


% (5) Solve the PEP
P.solve(verbose)

% (6) is the potential verified?
fprintf('Is the potential verified to tolerance? [0/1]: %d\n',double(expression_to_verify)<tolerance)


