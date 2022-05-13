function [worstcase] = OG(gamma,L,N,verbose)

% Consider the problem of finding a zero of a monotone Lipschitz operator:
%       find x such that F(x) = 0
% where F is monotone and L-Lipschitz.
%
% This function computes an upper bound on the worst-case performance of
% the optimistic gradient method (OG) for solving the monotone inclusion
% problems. This method is described by two sets of iterates: x^k, tx^k
% where k denotes the iteration counter, as follows: initialize
% tx^0 = tx^{-1} = tx^{-2} then run for k=1,...,N-1:
% tx^{k+1} = tx^k - 2 * \gamma * F(\tx^k) + \gamma * F(\tx^{k-1}),
%
%
% The code computes the worst-case ratio ||F(tx^N)||^2/||tx^0-x^*||^2,
% where x^* is a solution to the inclusion problem: F(x^*) = 0.
% This corresponds to computing the worst-case value of ||F(tx^N)||^2 when
% ||tx^0 - x^*||^2 == 1.

% (0) Initialize an empty PEP
P=pep();

% (1) Set up the problem class
paramF.L  =  L; paramF.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
F = P.DeclareFunction('LipschitzStronglyMonotone',paramF);

% (2) Set up the starting points
tx0 = P.StartingPoint(); % this is tx^0
xs  = F.OptimalPoint();  % this is some x^*

% Normalize the initial distance || tx^0 - x^* ||^2 == 1
P.InitialCondition( (tx0-xs)^2 == 1);

% (3) Algorithm
tx              = tx0;
current_F_tx    = F.gradient(tx0);   % this is F(tx^k)
last_F_tx       = current_F_tx;     % this is F(tx^{k-1})

for i = 1:N
    tx              = tx - 2* gamma * current_F_tx + gamma * last_F_tx;
    last_F_tx       = current_F_tx;
    current_F_tx    = F.gradient(tx);
end

% (4) Set up the performance measure:
P.PerformanceMetric( current_F_tx^2 );

% (5) Solve the PEP
P.solve(verbose);

% (6) Evaluate the output
worstcase = double(current_F_tx^2);

end
