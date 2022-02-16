clear all; clc;
%
% In this example, we consider Extragradient method:
% w_{k+1} = w_k - gamma_2 * F(w_k - gamma_1 * F(w_k))
% with gamma_2 = gamma_1 / 2
%

Nmax = 6;

for N = 6:Nmax
    % (0) Initialize an empty PEP
    P=pep();

    % (1) Set up the class of monotone inclusions
    param.L  =  1; param.mu = 0; % F is 1-Lipschitz and 0-strongly monotone

    gamma1 = 1/(2*param.L);
    gamma2 = gamma1;

    F = P.DeclareFunction('LipschitzStronglyMonotone',param);

    % (2) Set up the starting points
    w0=P.StartingPoint();
    [ws, Fs] = F.OptimalPoint(); 

    P.InitialCondition((ws-w0)^2<=1);  % Normalize the initial distance ||w0-ws||^2 <= 1

    % (3) Algorithm

    w = cell(N+1,1);
    w{1} = w0;
    for i = 1:N
        w12 = w{i} - gamma1 * F.evaluate(w{i});
        w{i+1} = w{i} - gamma2 * F.evaluate(w12);
    end

    % (4) Set up the performance measure: ||F(w_k)||^2
    squared_norm = (F.evaluate(w{N+1}))^2;
    P.PerformanceMetric(squared_norm);

    % (5) Solve the PEP
    P.solve()

    % (6) Evaluate the output
    double(squared_norm)   % worst-case squared norm
    
    res_norm = double(squared_norm);
    res_init_dist = double((ws-w0)^2);
    
    save(strcat('dump/EG_norm_L_1_N_', sprintf('%d_', N), sprintf('_%f_', gamma1), sprintf('%f', gamma2),'.mat'), 'res_norm', 'gamma1', 'gamma2', 'res_init_dist');

    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 = %f, ||x^0 - x^*||^2 = %f\n", double(squared_norm), res_init_dist);
    fprintf("======================================================\n");
end