#!/bin/bash
#SBATCH --job-name=mpi4py-test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=8               # total number of tasks
#SBATCH --cpus-per-task=1      # cpu-cores per task
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)
module load anaconda3
module load openmpi/gcc/3.1.3/64
#srun  python ../scripts/alf_build_model.py zsol_a+0.2
srun  python ../scripts/alf.py zsol_a+0.2
