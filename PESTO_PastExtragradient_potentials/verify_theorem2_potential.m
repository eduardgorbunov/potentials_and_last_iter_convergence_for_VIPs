%
% This code aims at verifying (numerically) the first inequality from
% Theorem 2 of the paper
%   "List-Iterate Convergence of Optimistic Gradient Method for Monotone
%       Variationnal Inequalities".
%
% PROBLEM SETUP:
% Consider the problem of finding a zero of a monotone Lipschitz operator:
%       find x\in Q such that F(x) = 0
% where F is monotone and L-Lipschitz, and Q is convex & compact.
%
% ALGORITHM: 
% The past extragradient method is described by two sets of iterates:
% x^k, tx^k where k denotes the iteration counter, as follows:
% initialize tx^0 = x^0 and x^1  = Proj_Q(x^0 - gamma * F(x^0)), then run
% for k=1,...,N-1:
%   tx^k    = Proj_Q(x^k - gamma * F(tx^{k-1}))
%   x^{k+1} = Proj_Q(x^k - gamma * F(tx^k))
%
% Denotes p_k := ||x^k - x^{k-1}||^2 
%               + || x^k - x^{k-1} - 2 gamma * (F(x^k) - F(tx^{k-1})) ||^2;
% the code compute the maximum (i.e., worst-case) value of
%   ||x^{k+1} - x^* ||^2 + 1/16 || tx^{k} - tx^{k-1} ||^2 
%           + A_{k+1} * p_{k+1}
%           - ||x^{k} - x^* ||^2 - 1/16 || tx^{k-1} - tx^{k-2} ||^2 
%           - A_k * p_k
% when A_{k+1} = A_k + 1/8. The expression should always be <= 0 for
% verifying the identity from Theorem 1 (with gamma<=1/4/L and A_k>=4/3).
% In the code below, we use k = 2 for notational convenience.
clear all; clc;
L = 1; gamma = 1/4/L; verbose = 2;
A2 = 20000;
tolerance = 1e-8;

% (0) Initialize an empty PEP
P=pep();

% (1) Set up the problem class
paramF.L  =  L; paramF.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
F = P.DeclareFunction('LipschitzStronglyMonotone',paramF);

xs  = F.OptimalPoint();  % this is some x^*

% (2) Set up the starting points
tx0 = P.StartingPoint(); % this is tx^01
x1  = P.StartingPoint(); % this is x^1

Ftx0    = F.gradient(tx0);
Fx1     = F.gradient(x1);
tx1     = x1 - gamma * Ftx0; 
Ftx1    = F.gradient(tx1);
x2      = x1 - gamma * Ftx1;
Fx2     = F.gradient(x2);
tx2     = x2 - gamma * Ftx1;
Ftx2    = F.gradient(tx2);
x3      = x2 - gamma * Ftx2;
Fx3     = F.gradient(x3);


p2 = (x2-x1)^2 + (x2-x1-2*gamma*(Fx2-Ftx1))^2;
p3 = (x3-x2)^2 + (x3-x2-2*gamma*(Fx3-Ftx2))^2;
A3 = A2 + 1/8;
expression1 = ( x2-xs )^2 + 1/16 * (tx1 - tx0)^2 + A2 * p2;
expression2 = ( x3-xs )^2 + 1/16 * (tx2 - tx1)^2 + A3 * p3;
% (4) Set up the performance measure:
expression_to_verify = expression2 - expression1;
P.PerformanceMetric(expression_to_verify);


% (5) Solve the PEP
P.solve(verbose)

% (6) is the potential verified?
fprintf('Is the potential verified to tolerance? [0/1]: %d\n',double(expression_to_verify)<tolerance)


