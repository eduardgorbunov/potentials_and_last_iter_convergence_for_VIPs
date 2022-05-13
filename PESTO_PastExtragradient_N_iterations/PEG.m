function [worstcase] = PEG(gamma,L,N,verbose)

% Consider the problem of finding a zero of a monotone Lipschitz operator:
%       find x such that F(x) = 0
% where F is monotone and L-Lipschitz.
%
% This function computes an upper bound on the worst-case performance of
% the past extragradient method (PEG) for solving the monotone inclusion
% problems. This method is described by two sets of iterates: x^k, tx^k
% where k denotes the iteration counter, as follows: initialize tx^0 = x^0
% and x^1  = x^0 - gamma * F(x^0), then run for k=1,...,N-1:
%   tx^k    = x^k - gamma * F(tx^{k-1})
%   x^{k+1} = x^k - gamma * F(tx^k)
%
% The code computes the worst-case ratio ||F(x^N)||^2/||x^0-x^*||^2,
% where x^* is a solution to the inclusion problem: F(x^*) = 0.
% This corresponds to computing the worst-case value of ||F(x^N)||^2 when
% ||x^0 - x^*||^2 == 1.

% (0) Initialize an empty PEP
P=pep();

% (1) Set up the problem class
paramF.L  =  L; paramF.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
F = P.DeclareFunction('LipschitzStronglyMonotone',paramF);

% (2) Set up the starting points
x0 = P.StartingPoint(); % this is x^0
xs = F.OptimalPoint();  % this is some x^*

% Normalize the initial distance || x^0 - x^* ||^2 == 1
P.InitialCondition( (x0-xs)^2 == 1);

% (3) Algorithm
tx = x0;
x  = x0;
current_F_tx = F.gradient(tx); % this is the last evaluation of F(tx^k)

for i = 1:N
    tx              = x - gamma * current_F_tx;
    current_F_tx    = F.gradient(tx);
    x               = x - gamma * current_F_tx;
    % we also evaluate F(x^k) (although not used in the actual algo)
    current_F_x     = F.gradient(x);
end

% (4) Set up the performance measure:
P.PerformanceMetric( current_F_x^2 );

% (5) Solve the PEP
P.solve(verbose);

% (6) Evaluate the output
worstcase = double(current_F_x^2);

end
