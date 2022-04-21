clear all; clc;
%
% In this example, we consider Optimistic gradient method:
% w_{k+1} = w_k - 2 * gamma * F(w_k) + gamma * F(w_{k-1})

%

Nmax = 40;
alphas = {1.0/3, 1.0/4, 1.0/5};

for index = 2:3
    for N = 1:Nmax
    % (0) Initialize an empty PEP
    P=pep();

    % (1) Set up the class of monotone inclusions
    param.L  =  1; param.mu = 0; % F is 1-Lipschitz and 0-strongly monotone
    %param.beta = 1.0;

    gamma = alphas{index}*1.0/param.L;
    %gamma = param.beta/3;

    F = P.DeclareFunction('LipschitzStronglyMonotone',param);
    %F = P.DeclareFunction('Cocoercive',param);

    % (2) Set up the starting points
    w0=P.StartingPoint();
    [ws, Fs] = F.OptimalPoint(); 

    P.InitialCondition((ws-w0)^2<=1);  % Normalize the initial distance ||w0-ws||^2 <= 1

    % (3) Algorithm

    w = cell(N+1,1);
    tw = cell(N+1,1);
    w{1} = w0;
    tw{1} = w0;
    g_prev = F.evaluate(tw{1});
    w{2} = w0 - gamma * g_prev;
    for i = 2:N
        tw{i} = w{i} - gamma * g_prev;
        g_current = F.evaluate(tw{i});
        w{i+1} = w{i} - gamma * g_current;
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
    
    save(strcat('dump/EFTP_norm_diff_Lip_l_1_N_', sprintf('%d_', N), sprintf('_%f', gamma),'.mat'), 'res_norm', 'gamma', 'res_init_dist');

    fprintf("======================================================\n");
    fprintf("N = %d: ", N);
    fprintf("||F(x^N)||^2 - ||F(x^{N-1})||^2 = %f, ||x^0 - x^*||^2 = %f\n", double(squared_norm), res_init_dist);
    fprintf("======================================================\n");
end
end