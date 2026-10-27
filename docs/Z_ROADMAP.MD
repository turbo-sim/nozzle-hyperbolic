


The prompt to create the 2 examples worked



------
We have to establish baseline for the different FVS and FDS methods
check also the different time integration methods
Check first order and MUSCL


-------

We have to implement a one-shot field inversion using slsqp for the HEM approach and see if it Is possible to solve everything at the same time

Compare to adjoint based optimization.

Should the optimistix solvers8 be adapted somehow for the initial step size and rejection of steps that are unfeasible? Would things still be differentiable? perhaps we can use our custom robust solver and then solve with optimistix once that we are close to the actual solution?