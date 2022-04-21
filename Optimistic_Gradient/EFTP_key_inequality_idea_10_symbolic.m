clc; clear all;
verbose = 0;
%
%   find x^* such that F(x^*) = 0   with F: L-Lip. and monotone
%
%   tx^{k+1} = x^k - \gamma_k g^k  with g^k = F(tx^k)
%   x^{k+1} = x^k - gamma_k tg^k    with tg^k = F(tx^{k+1})
%
%
%   we study worst-case behavior of \|F(x^N)\|^2  (\|g^N\|^2)
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x0 tg0 g0 tg1 g1 ... tgN gN ]   and G = P^T P
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

Nmax = 2;

for N = Nmax:Nmax
    L = 1;
    R = 1; % this is the bound on ||x_0-x_*||^2

    gamma = 1/(3*L);

    % internal notation:

    dimG    = 2*N + 2;        % dimension of G
    nbPts   = 2*N + 2; % this is x_*, x_0, tx1, x_1, tx_2, x_2 ..., tx_N, x_N

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
%     G = sdpvar(dimG); % this defines a symmetric matrix variable, called G
    lambda_Lipschitz = sym('lambda_Lipsch%d%d', [nbPts nbPts]);%there will be many unused ones, but convenient notation
    lambda_monotone  = sym('lambda_monot%d%d', [nbPts nbPts]);%there will be many unused ones, but convenient notation
    tau = sym('tau'); % this is the multiplier for ||x_0-x_*||^2 <= R^2
    
%     constraint = ( G >= 0); % G is PSD
%     constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

    dual_PSD_matrix = tau * (barx0-barxstar)*(barx0-barxstar)' - (bargk(:,end))*(bargk(:,end))' - 2*(bargk(:,end)-bargk(:,end-1))*(bargk(:,end)-bargk(:,end-1))' + (bargk(:,end-2))*(bargk(:,end-2))' + 2*(bargk(:,end-2)-bargk(:,end-3))*(bargk(:,end-2)-bargk(:,end-3))';
    objective = tau;
    
%     objective = (bargk(:,end))'*G*(bargk(:,end)); % this is \|F(x^N)\|^2

    
    % this is for the constraints on F
    for i = 2:nbPts
        for j = 1:i
            if i~=j & ((i ~= 3)) & (i - j <= 2) & (j ~= 1) & (i == nbPts)
%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barg(:,i) - barg(:,j)) - L^2 * (barx(:,i) - barx(:,j))'*G*(barx(:,i) - barx(:,j)) <= 0);
                %  \|g^i - g^j\|^2 \leq L^2*\|x^i - x^j\|
                A = (barg(:,i) - barg(:,j))*(barg(:,i) - barg(:,j))'-L^2 * (barx(:,i) - barx(:,j))*(barx(:,i) - barx(:,j))';
                A = 1/2 * (A+A');
                dual_PSD_matrix = dual_PSD_matrix + lambda_Lipschitz(i,j) * A;
                

%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
                %  <g^i - g^j, x^i - x^j> >= 0
                A = (barg(:,i) - barg(:,j))*(barx(:,i) - barx(:,j))';
                A = 1/2 * (A+A');
                dual_PSD_matrix = dual_PSD_matrix - lambda_monotone(i,j) * A; % minus because of " ... >= 0 "
            end
        end
    end
    %
    
end