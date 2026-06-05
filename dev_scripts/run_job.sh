#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job --
#BSUB -J NozzleHyperbolic
# -- specify queue --
#BSUB -q hpc
# -- estimated wall clock time (execution time): hh:mm --
#BSUB -W 00:30
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=12GB]"
# -- number of processors
#BSUB -n 6
# --specify that the cores MUST BE on a single host! --
#BSUB -R "span[hosts=1]" 
# -- user email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u alber@dtu.dk
# -- mail notification --
# -- at start --
#BSUB -B
# -- at completion --
#BSUB -N
# --Specify the output and error file. %J is the job-id --
# --  -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo nozzle_hyperbolic_%J.out
#BSUB -eo nozzle_hyperbolic_%J.err

# Load poetry
source ../.venv/bin/activate

# Let XLA use all allocated cores for both compilation and runtime.
# Compilation in particular is single-threaded by default — this cuts it
# significantly on the first run (and is a no-op once the cache is warm).
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$LSB_DJOB_NUMPROC"
export JAX_TRACEBACK_FILTERING=off   # full tracebacks on cluster errors

# Run
poetry run python main.py
