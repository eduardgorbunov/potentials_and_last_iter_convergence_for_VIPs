clear all; clc;

L = 1;
N_list = [1:10 12:4:40];
verbose = 0;

nb_trials = length(N_list);
PEG_wc4 = zeros(1,nb_trials);
OG_wc4  = zeros(1,nb_trials);
th_H0   = zeros(1,nb_trials);
for i = 1:nb_trials
    tic;
    PEG_wc4(i) = Proj_OG(1/4/L,L,N_list(i),verbose);
    OG_wc4(i) = Proj_PEG(1/4/L,L,N_list(i),verbose);
    th_H0(i) = 24./(3*N_list(i)+32);
    timing = toc;
    fprintf('%d done on %d [time: %5.4f sec.]\n',i,nb_trials,timing);
end

close all; figure;
loglog(N_list,PEG_wc4,'-r','linewidth',2);  hold on;
loglog(N_list,OG_wc4,'-b','linewidth',2);
loglog(N_list,5./N_list,'-c','linewidth',2);
loglog(N_list,PEG_wc4,'.r','linewidth',2);  hold on;
loglog(N_list,OG_wc4,'.b','linewidth',2);
xlabel('Iteration'); ylabel('Residual norm');
legend('PEG (1/4/L)', 'OG (1/4/L)', '5/k');
save('constrained.mat','N_list','L','PEG_wc4','OG_wc4');

