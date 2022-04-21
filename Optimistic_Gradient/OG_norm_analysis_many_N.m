clear all; clc;
%
% In this example, we consider Optimistic gradient method:
% w_{k+1} = w_k - 2 * gamma * F(w_k) + gamma * F(w_{k-1})

%

Nmax = 40;

for N = 1:Nmax
    % (0) Initialize an empty PEP
    P=pep();

    % (1) Set up the class of monotone inclusions
    %param.L  =  1; param.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
    param.beta = 1.0;

    %gamma = 1/(3*param.L);
    gamma = param.beta/5;

    %F = P.DeclareFunction('LipschitzStronglyMonotone',param);
    F = P.DeclareFunction('Cocoercive',param);

    % (2) Set up the starting points
    w0=P.StartingPoint();
    [ws, Fs] = F.OptimalPoint(); 

    P.InitialCondition((ws-w0)^2<=1);  % Normalize the initial distance ||w0-ws||^2 <= 1

    % (3) Algorithm

    w = cell(N+1,1);
    w{1} = w0;
    g_prev = F.evaluate(w{1});
    w{2} = w0 - gamma * g_prev;
    for i = 2:N
        g_current = F.evaluate(w{i});
        w{i+1} = w{i} - 2 * gamma * g_current + gamma * g_prev;
        g_prev = g_current;
    end

    % (4) Set up the performance measure: ||F(w_k)||^2
    squared_norm = (F.evaluate(w{N+1}))^2 - (F.evaluate(w{N}))^2;
    P.PerformanceMetric(squared_norm);

    % (5) Solve the PEP
    P.solve()

    % (6) Evaluate the output
    double(squared_norm)   % worst-case squared norm
    
    res_norm = double(squared_norm);
    res_init_dist = double((ws-w0)^2);
    
    save(strcat('dump/OG_norm_diff_l_1_N_', sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma', 'res_init_dist');

    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(tx^N)||^2 - ||F(tx^{N-1})||^2 = %f, ||x^0 - x^*||^2 = %f\n", double(squared_norm), res_init_dist);
    fprintf("======================================================\n");
end