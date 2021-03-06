clc; clear all;
verbose = 0;
%
%   find x^* such that F(x^*) = 0   with F: L-Lip. and monotone
%
%   tx^{k}  = x^k - gamma_k tg^{k-1}  with tg^{k-1} = F(tx^{k-1})
%   x^{k+1} = x^k - gamma_k tg^k      with tg^k     = F(tx^{k})
%
%
%   we study worst-case behavior of \|F(x^N)\|^2  (\|g^N\|^2)
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x^0 g^0 tg^0 g^1 tg^1 ... g^N ]   and G = P^T P
%
%   Define a few things for simplifying the code:
%   - \bar{x}^k :  x^k = P \bar{x}^k   (with k\in\{*,0,...N\})
%   - \bar{g}^k :  g^k = P \bar{g}^k
%   - \bar{tx}^k :  tx^k = P \bar{tx}^k   (with k\in\{*,0,...N\})
%   - \bar{tg}^k :  tg^k = P \bar{tg}^k
%  why this choice?
%  first: <g^i;x^k> = \bar{g}^i^T P^T P \bar{x}^k
%                   = \bar{g}^i^T G \bar{x}^k
%                   = Trace (G \bar{x}^k \bar{g}^i^T )
%  second:
%       \bar{x}^{k+1} = \bar{x}^k - gamma_k \bar{tg}^k
%

% algorithm setup:

Nmax = 30;

for N = 2:Nmax
    L = 1;
    R = 1; % this is the bound on ||x_0-x_*||^2

    gamma = 1/(6*L);

    % internal notation:

    dimG    = 2*N + 2;        % dimension of G
    nbPts   = 2*N + 2; % this is x^*, x^0, tx^0, x^1, tx^1, ..., tx^{N-1}, x^N

    barxstar = zeros(dimG,1);
    bargstar = zeros(dimG,1);

    barx0 = zeros(dimG, 1); barx0(1) = 1;
    
    bargk = zeros(dimG, 2*N+1); bargk(2:end,:) = eye(2*N+1); % so that bargk(:,i) returns gradient of x_{i-1}
    barxk = zeros(dimG, 2*N+1); barxk(:,1) = barx0; barxk(:,2) = barx0;
    barxk(:,3) = barxk(:,1) - gamma * bargk(:,2);
    
    for i = 2:N
        barxk(:,2*i) = barxk(:,2*i-1) - gamma * bargk(:,2*i-2);
        barxk(:,2*i+1) = barxk(:,2*i-1) - gamma * bargk(:,2*i); 
    end
    
    
    barx = [barxstar barxk];
    barg = [bargstar bargk];

    %% SDP

    % this uses YALMIP:
    G = sdpvar(dimG); % this defines a symmetric matrix variable, called G

    constraint = ( G >= 0); % G is PSD
    constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

    objective = (bargk(:,end-1))'*G*(bargk(:,end-1)) - (bargk(:,end-3))'*G*(bargk(:,end-3)); % this is \|F(tx^N)\|^2 - \|F(tx^{N-1})\|^2

    % this is for the constraints on F
    for i = 2:nbPts
        for j = 1:i
            if i~=j & ((i ~= 3) | (i == 3 & j == 1)) 
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
    
    res_norm = double(objective);
    save(strcat('dump/EFTP_tilde_norm_diff_L_1_N_', sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma');
    
    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 - ||F(x^{N-1})||^2 = %f\n", res_norm);
    fprintf("======================================================\n");
end