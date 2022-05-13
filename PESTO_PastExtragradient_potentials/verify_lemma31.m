%
% This code aims at verifying (numerically) the inequality from
% Lemma 3.1 of the paper
%   "List-Iterate Convergence of Optimistic Gradient Method for Monotone
%       Variationnal Inequalities".
%
% PROBLEM SETUP:
% Consider the problem of finding a zero of a monotone Lipschitz operator:
%       find x such that F(x) = 0
% where F is monotone and L-Lipschitz.
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
%   p_{k+1} - p_k - 3 ( L^2 gamma^2 - 2/9) || F(tx^k) - F(tx^{k-1}) ||^2
% which should always be <= 0 for verifying the identity from Lemma 3.1.
% In the code below, we use k = 1 for notational convenience.

clear all; clc;

% parameters
L = 1; gamma = 10;

verbose = 1;
tolerance = 1e-6;

% (0) Initialize an empty PEP
P=pep();

% (1) Set up the problem class
paramF.L  =  L; paramF.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
F = P.DeclareFunction('LipschitzStronglyMonotone',paramF);

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
residual = 3 * (L*gamma^2 - 2/9) * ( Ftx1 - Ftx0 )^2;
% (4) Set up the performance measure:
expression = p2 - p1 - residual;
P.PerformanceMetric(expression_to_verify);

% (5) Solve the PEP
P.solve(verbose)

% (6) is the potential verified?
fprintf('Is the potential verified to tolerance? [0/1]: %d\n',double(expression)<tolerance)
