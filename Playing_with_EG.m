clc; clear all;
verbose = 1;
%
%   find x^* such that F(x^*) = 0   with F: L-Lip. and monotone
%
%   y^k = x^k - \gamma_k g^k  with g^k = F(x^k)
%   x^{k+1} = x^k - gamma_k h^k    with h^k = F(y^k)
%
%
%   we study worst-case behavior of \|F(x^N)\|^2  (\|g^N\|^2)
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x0 g0 g1 ... gN ]   and G = P^T P
%
%   Define a few things for simplifying the code:
%   - \bar{x}^k :  x^k = P \bar{x}^k   (with k\in\{*,0,...N\})
%   - \bar{g}^k :  g^k = P \bar{g}^k
%  why this choice?
%  first: <g^i;x^k> = \bar{g}^i^T P^T P \bar{x}^k
%                   = \bar{g}^i^T G \bar{x}^k
%                   = Trace (G \bar{x}^k \bar{g}^i^T )
%  second:
%       \bar{x}^{k+1} = \bar{x}^k - 2gamma_k \bar{g}^k + \gamma_k g^{k-1}
%

% algorithm setup:
N = 64;
L = 1;
R = 1; % this is the bound on ||x_0-x_*||^2

gamma = @(k)(1/(4*L));

% internal notation:

dimG    = N + 2;        % dimension of G
nbPts   = N + 2; % this is x_*, x_0, ..., x_N

barxstar = zeros(dimG,1);
bargstar = zeros(dimG,1);

barx0 = zeros(dimG, 1); barx0(1) = 1;

bargk = zeros(dimG, N+1); bargk(2:end,:) = eye(N+1); % so that bargk(:,i) returns gradient of x_{i-1}
barxk = zeros(dimG, N+1); barxk(:,1) = barx0;

ykminus1 = barx0; % this is for accelarated version
barxk(:,2) = barxk(:,1) - gamma(1) * bargk(:,1);
for i = 2:N
    barxk(:,i+1) = barxk(:,i) - 2*gamma(i) * bargk(:,i) + gamma(i) * bargk(:,i-1);
    %     yk           = barxk(:,i) - gamma * bargk(:,i);
    %     barxk(:,i+1) = yk + (i-1)/(i+2) * (yk-ykminus1);
    %     ykminus1 = yk;
end


barx = [barxstar barxk];
barg = [bargstar bargk];

%% SDP

% this uses YALMIP:
G = sdpvar(dimG); % this defines a symmetric matrix variable, called G

constraint = ( G >= 0); % G is PSD
constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

objective = (bargk(:,end))'*G*(bargk(:,end)); % this is \|F(x^N)\|^2

% this is for the constraints on F
for i = 2:nbPts
    for j = 1:i
        if i~=j & (i - j <= 2 | j == 1)
            constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barg(:,i) - barg(:,j)) - L^2 * (barx(:,i) - barx(:,j))'*G*(barx(:,i) - barx(:,j)) <= 0);
            %  \|g^i - g^j\|^2 \leq L^2*\|x^i - x^j\|
            constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
            %  <g^i - g^j, x^i - x^j> >= 0
        end
    end
end
%

options = sdpsettings('verbose',verbose);
optimize(constraint,-objective,options);

[double(objective)]