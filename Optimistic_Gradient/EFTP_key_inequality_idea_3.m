clc; clear all;
verbose = 0;
%
%   find x^* such that F(x^*) = 0   with F: L-Lip. and monotone
%
%   tx^{k+1} = x^k - \gamma_k g^k  with g^k = F(tx^k)
%   x^{k+1} = x^k - gamma_k tg^k    with tg^k = F(tx^{k+1})
%   hx^{k+1} = x^k - gamma_k hg^k with hg^k = F(tx^{k+1} - gamma*tg^k)
%
%   we study worst-case behavior of \|F(x^N)\|^2  (\|g^N\|^2)
%   under the condition that ||x_0 - x_*|| is bounded.
%
%
%   Let's PEP-it!
%   - P = [ x0 tg0 g0 ttg0 hg0 tg1 g1 ttg1 hg1 ... tgN gN ttgN hgN]   and G = P^T P
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

Nmax = 55;

for N = 1:Nmax
    L = 1;
    R = 1; % this is the bound on ||x_0-x_*||^2

    gamma = 1/(4*L);

    alpha = gamma;

    % internal notation:

    dimG    = 4*N + 2;        % dimension of G
    nbPts   = 4*N + 2; % this is x_*, x_0, tx1, x_1, ttx_1, hx_1, tx_2, x_2, ttx_2, hx_2 ..., tx_N, x_N, ttx_N+1, hx_N+1

    barxstar = zeros(dimG,1);
    bargstar = zeros(dimG,1);

    barx0 = zeros(dimG, 1); barx0(1) = 1;
    
    bargk = zeros(dimG, 4*N+1); bargk(2:end,:) = eye(4*N+1); % so that bargk(:,i) returns gradient of x_{i-1}
    barxk = zeros(dimG, 4*N+1); barxk(:,1) = barx0; barxk(:,2) = barx0;
    barxk(:,3) = barxk(:,1) - gamma * bargk(:,2); % x_1
    barxk(:,4) = barxk(:,2) - alpha * bargk(:,2); % ttx_1
    barxk(:,5) = barxk(:,1) - alpha * bargk(:,4); % hx_1
    
    for i = 2:N
        barxk(:,4*i-2) = barxk(:,4*i-5) - gamma * bargk(:,4*i-6);
        barxk(:,4*i-1) = barxk(:,4*i-5) - gamma * bargk(:,4*i-2);
        barxk(:,4*i) = barxk(:,4*i-2) - alpha * bargk(:,4*i-2);
        barxk(:,4*i+1) = barxk(:,4*i-3) - alpha * bargk(:,4*i);
    end
    
    barx = [barxstar barxk];
    barg = [bargstar bargk];

    %% SDP

    % this uses YALMIP:
    G = sdpvar(dimG); % this defines a symmetric matrix variable, called G

    constraint = ( G >= 0); % G is PSD
    constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

   
    objective = (bargk(:,end))'*G*(bargk(:,end)) - (bargk(:,end-4))'*G*(bargk(:,end-4));

    % this is for the constraints on F
    for i = 2:nbPts
        for j = 1:i
            if i~=j & (i ~= 3) & (i - j <= 100000) & (i ~= 5 | j ~= 4) & (j ~= 1)
                if true
                    constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barg(:,i) - barg(:,j)) - L^2 * (barx(:,i) - barx(:,j))'*G*(barx(:,i) - barx(:,j)) <= 0);
                    %  \|g^i - g^j\|^2 \leq L^2*\|x^i - x^j\|
                end
                if true
                    constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
                    %  <g^i - g^j, x^i - x^j> >= 0
                end
                %constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
                    %  <g^i - g^j, x^i - x^j> >= 0
            end
        end
    end
    %
    
    options = sdpsettings('verbose',verbose);
    optimize(constraint,-objective,options);

    [double(objective)]
    
    res_norm = double(objective);
    %save(strcat('dump/EFTP_key_ineq_simple_dist_2_res_L_1_N_', sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma');
    
    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 - sum = %f\n", res_norm);
    fprintf("======================================================\n");

    %fprintf("Dual variables\n");
    index_constraints = 2;
    monotonicity_weights = [];
    Lipschitzness_weights = [];
    for i = 2:nbPts
        for j = 1:i
            if i~=j & (i ~= 3) & (i - j <= 100000) & (i ~= 5 | j ~= 4) & (j ~= 1)
                if true
                    index_constraints = index_constraints + 1;
                    %fprintf("Lipschitzness at (%d, %d): %f\n", i, j, dual(constraint(index_constraints)));
                    Lipschitzness_weights = [Lipschitzness_weights, dual(constraint(index_constraints))];
                end
                if true
                    index_constraints = index_constraints + 1;
                    %fprintf("Monotonicity  at (%d, %d): %f\n", i, j, dual(constraint(index_constraints)));
                    monotonicity_weights = [monotonicity_weights, dual(constraint(index_constraints))];
                end
            end
        end
    end
    %save(strcat('dump/EFTP_key_ineq_simple_dist_2_dual_variables_L_', sprintf('%d_', L), sprintf('N_%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma', 'monotonicity_weights', 'Lipschitzness_weights');
    %fprintf("======================================================\n");
end