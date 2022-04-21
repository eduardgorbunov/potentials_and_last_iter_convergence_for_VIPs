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

Nmax = 70;

for N = 1:Nmax
    L = 1;
    R = 1; % this is the bound on ||x_0-x_*||^2

    gamma = 1/(4*L);

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
    lambda_Lipschitz = sdpvar(nbPts,nbPts,'full');%there will be many zeros inside, but simple for notational convenience!
    lambda_monotone  = sdpvar(nbPts,nbPts,'full');%there will be many zeros inside, but simple for notational convenience!
    tau = sdpvar(1); % this is the multiplier for ||x_0-x_*||^2 <= R^2
    
%     constraint = ( G >= 0); % G is PSD
%     constraint = constraint + ( (barx0-barxstar)'*G*(barx0-barxstar) <= R^2);  % this is ||x_0-x_*||^2 <= R^2

    dual_PSD_matrix = tau * (barx0-barxstar)*(barx0-barxstar)' - (bargk(:,end))*(bargk(:,end))';
    objective = tau;
    
%     objective = (bargk(:,end))'*G*(bargk(:,end)); % this is \|F(x^N)\|^2
    
    constraint = (lambda_Lipschitz>=0);
    constraint = constraint + (lambda_monotone>=0);
    constraint = constraint + (lambda_Lipschitz(1,1)==0);
    constraint = constraint + (lambda_monotone(1,1)==0);
    % this is for the constraints on F
    for i = 2:nbPts
        for j = 1:i
            if i~=j & ((i ~= 3) | (i == 3 & j == 1)) & (i - j <= 2 | j == 1)
%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barg(:,i) - barg(:,j)) - L^2 * (barx(:,i) - barx(:,j))'*G*(barx(:,i) - barx(:,j)) <= 0);
                %  \|g^i - g^j\|^2 \leq L^2*\|x^i - x^j\|
                if ((mod(i,2) == 0) & (j == i-1) & (j ~= 1))% | ((mod(i,2) == 1) & (j == i-2) & (j ~= 1))
                    A = (barg(:,i) - barg(:,j))*(barg(:,i) - barg(:,j))'-L^2 * (barx(:,i) - barx(:,j))*(barx(:,i) - barx(:,j))';
                    A = 1/2 * (A+A');
                    dual_PSD_matrix = dual_PSD_matrix + lambda_Lipschitz(i,j) * A;
                    constraint = constraint + (lambda_Lipschitz(j,i)==0);
                    if ((i >= 6) & (i < nbPts))
                        constraint = constraint + (lambda_Lipschitz(i,j) - lambda_Lipschitz(i-2,j-2) == lambda_Lipschitz(i-2,j-2) - lambda_Lipschitz(i-4,j-4));
                    end
                else
                    constraint = constraint + (lambda_Lipschitz(i,j)==0);
                    constraint = constraint + (lambda_Lipschitz(j,i)==0);
                end
                
%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
                %  <g^i - g^j, x^i - x^j> >= 0
                if ((mod(i,2) == 1) & (j == 1)) | ((mod(i,2) == 0) & (j == i-2) & (j ~= 1)) | ((i == 4) & (j == 1))
                    A = (barg(:,i) - barg(:,j))*(barx(:,i) - barx(:,j))';
                    A = 1/2 * (A+A');
                    dual_PSD_matrix = dual_PSD_matrix - lambda_monotone(i,j) * A; % minus because of " ... >= 0 "
                    constraint = constraint + (lambda_monotone(j,i)==0);
                    if ((mod(i,2) == 1) & (j == 1) & (i > 3) & (i < nbPts-1))
                        constraint = constraint + (lambda_monotone(i,j)==lambda_monotone(i-2,j));
                    end
                else
                    constraint = constraint + (lambda_monotone(i,j)==0);
                    constraint = constraint + (lambda_monotone(j,i)==0);
                end
            else
                constraint = constraint + (lambda_Lipschitz(i,j)==0);
                constraint = constraint + (lambda_Lipschitz(j,i)==0);
                constraint = constraint + (lambda_monotone(i,j)==0);
                constraint = constraint + (lambda_monotone(j,i)==0);
            end
        end
    end

    %
    constraint = constraint + (dual_PSD_matrix>=0);
    
    options = sdpsettings('verbose',verbose);
    optimize(constraint,objective,options);  % we minimize objectiv =tau <-- 

    [double(objective)]
    
    res_norm = double(objective);
    %save(strcat('dump/EFTP_2_simplified_dual_8_2_norm_L_', sprintf('%d_N_', L), sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma');
    
    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 = %f\n", res_norm);
    fprintf("======================================================\n");

    monotonicity_weights = [];
    Lipschitzness_weights = [];

    for i = 2:nbPts
        for j = 1:i
            if i~=j & ((i ~= 3) | (i == 3 & j == 1)) & (i - j <= 2 | j == 1)
%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barg(:,i) - barg(:,j)) - L^2 * (barx(:,i) - barx(:,j))'*G*(barx(:,i) - barx(:,j)) <= 0);
                %  \|g^i - g^j\|^2 \leq L^2*\|x^i - x^j\|
                if ((mod(i,2) == 0) & (j == i-1) & (j ~= 1))% | ((mod(i,2) == 1) & (j == i-2) & (j ~= 1))
                    Lipschitzness_weights = [Lipschitzness_weights, double(lambda_Lipschitz(i,j))];
                end
                
%                 constraint = constraint + ( (barg(:,i) - barg(:,j))'*G*(barx(:,i) - barx(:,j)) >= 0);
                %  <g^i - g^j, x^i - x^j> >= 0
                if ((mod(i,2) == 1) & (j == 1)) | ((mod(i,2) == 0) & (j == i-2) & (j ~= 1)) | ((i == 4) & (j == 1))
                    monotonicity_weights = [monotonicity_weights, double(lambda_monotone(i,j))];
                end
            end
        end
    end
    %save(strcat('dump/EFTP_2_simplified_dual_8_2_dual_var_L_', sprintf('%d_', L), sprintf('N_%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma', 'monotonicity_weights', 'Lipschitzness_weights');
    %fprintf("Dual variables\n");
    %fprintf("\t Lipschitzness:\n")
    %double(lambda_Lipschitz)
    %fprintf("\t monotonicity:\n")
    %double(lambda_monotone)
    
    %fprintf("\t PSD matrix:\n")
    %double(dual_PSD_matrix)
end