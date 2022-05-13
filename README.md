# Last-iterate convergence of optimistic gradient

This code comes jointly with reference

> [1] "Last-Iterate Convergence of Optimistic Gradient Method for Monotone Variational Inequalities".

Date:    May 2022

## Requirements

**Packages.** To run our MATLAB code one should install Performance Estimation Toolbox (PESTO) https://github.com/AdrienTaylor/Performance-Estimation-Toolbox, SEDUMI https://yalmip.github.io/solver/sedumi/ (or any other appropriate SDP solver), YALMIP https://yalmip.github.io/ and add the to the path when executing our code.

The Python codes for visualization require Jupyter notebooks.

## Organization of the code

The code is divided in three parts:
- PESTO codes for directly assessing the worst-case convergence speed of the methods under consideration, see folder [PESTO_PastExtragradient_N_iterations](/PESTO_PastExtragradient_N_iterations)).
- PESTO codes for verifying numerically Lemma 3.1, Lemma 4.1 as well as the potential functions from Theorem 1 and Theorem 2, see folder [PESTO_PastExtragradient_potentials](/PESTO_PastExtragradient_potentials)).
- YALMIP codes for playing with the SDP formulations (less readable than PESTO codes but allows to go into the details), see folder [DirectSDPs_Codes](/DirectSDPs_Codes).


