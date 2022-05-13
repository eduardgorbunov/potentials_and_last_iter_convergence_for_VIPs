clear all; clc;

L = 1;
N_list = [1:10 12:4:40];
verbose = 0;

nb_trials = length(N_list);
PEG_wc3 = zeros(1,nb_trials);
OG_wc3 = zeros(1,nb_trials);
theory_wc3 = zeros(1,nb_trials);
PEG_wc4 = zeros(1,nb_trials);
OG_wc4 = zeros(1,nb_trials);
theory_wc4 = zeros(1,nb_trials);

th_bound = @(gamma,N)(3*(1+32*L^2*gamma^2)/gamma^2/(N+32));
for i = 1:nb_trials
    tic;
    PEG_wc3(i) = PEG(1/3/L,L,N_list(i),verbose);
    OG_wc3(i) = OG(1/3/L,L,N_list(i),verbose);
    PEG_wc4(i) = PEG(1/4/L,L,N_list(i),verbose);
    OG_wc4(i) = OG(1/4/L,L,N_list(i),verbose);
    theory_wc3(i) = th_bound(1/3/L,N_list(i));
    theory_wc4(i) = th_bound(1/4/L,N_list(i));
    timing = toc;
    fprintf('%d done on %d [time: %5.4f sec.]\n',i,nb_trials,timing);
end
close all; figure;
loglog(N_list,PEG_wc3,'-r','linewidth',2);  hold on; 
loglog(N_list,OG_wc3,'-b','linewidth',2); 
loglog(N_list,theory_wc3,'-k','linewidth',2);  
loglog(N_list,PEG_wc4,'--r','linewidth',2);  
loglog(N_list,OG_wc4,'--b','linewidth',2); 
loglog(N_list,theory_wc4,'--k','linewidth',2); 
loglog(N_list,5./N_list,'-c','linewidth',2); 
loglog(N_list,PEG_wc3,'.r','linewidth',2);  hold on; 
loglog(N_list,OG_wc3,'.b','linewidth',2); 
loglog(N_list,PEG_wc4,'.r','linewidth',2);  
loglog(N_list,OG_wc4,'.b','linewidth',2); 
xlabel('Iteration'); ylabel('Operator norm');
legend('PEG (1/3/L)', 'OG (1/3/L)', 'Theory (1/3/L)','PEG (1/4/L)', 'OG (1/4/L)', 'Theory (1/4/L)');
save('unconstrained.mat','N_list','L','PEG_wc3','PEG_wc4','OG_wc3','OG_wc4','theory_wc3','theory_wc4');