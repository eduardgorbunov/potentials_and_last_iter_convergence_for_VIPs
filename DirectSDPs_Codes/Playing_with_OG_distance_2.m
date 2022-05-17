clc; clear all;
verbose = 0;
%
%   find x^* such that F(x^*) = 0   with F: L-Lip. and monotone
%
%   x^1 = x^0 - gamma_0 g^0
%   x^{k+1} = x^k - 2gamma_k g^k + gamma_k g^{k-1}   with g^k = F(x^k)
%
%
%   we study worst-case behavior of \|F(x^N)\|^2  (\|g^N\|^2)
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x^0 g^0 g^1 ... g^N ]   and G = P^T P
%
%   Define a few things for simplifying the code:
%   - \bar{x}^k :  x^k = P \bar{x}^k   (with k\in\{*,0,...N\})
%   - \bar{g}^k :  g^k = P \bar{g}^k
%  why this choice?
%  first: <g^i;x^k> = \bar{g}^i^T P^T P \bar{x}^k
%                   = \bar{g}^i^T G \bar{x}^k
%                   = Trace (G \bar{x}^k \bar{g}^i^T )
%  second:
%       \bar{x}^{k+1} = \bar{x}^k - 2gamma_k \bar{g}^k + gamma_k g^{k-1}
%

Nmax = 75;

for N = Nmax:Nmax
    % algorithm setup:
    L = 1;
    R = 1; % this is the bound on ||x_0-x_*||^2

    gamma = 1/(3*L);

    % internal notation:

    dimG    = N + 2;        % dimension of G
    nbPts   = N + 2; % this is x^*, x^0, ..., x^N

    barxstar = zeros(dimG,1);
    bargstar = zeros(dimG,1);

    barx0 = zeros(dimG, 1); barx0(1) = 1;

    bargk = zeros(dimG, N+1); bargk(2:end,:) = eye(N+1); % so that bargk(:,i) returns gradient of x_{i-1}
    barxk = zeros(dimG, N+1); barxk(:,1) = barx0;

    barxk(:,2) = barxk(:,1) - gamma * bargk(:,1);
    for i = 2:N
        barxk(:,i+1) = barxk(:,i) - 2*gamma * bargk(:,i) + gamma * bargk(:,i-1);
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
            if i~=j & ((i - j <= 2) | j == 1)
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
    
    double(objective)
    
    res_norm = double(objective);
    save(strcat('dump/OG_dist_2_norm_L_1_N_', sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma');
    
    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 = %f\n", res_norm);
   end