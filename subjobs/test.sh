#!/bin/bash
#SBATCH --job-name=mpi4py-test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1               # total number of tasks
#SBATCH --cpus-per-task=32               # total number of tasks
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:30:00          # total run time limit (HH:MM:SS)
module load anaconda3
module load openmpi/gcc/3.1.3/64

#srun  python ../scripts/alf_build_model.py ldss3_dr269_n1600_Re8_wave6e
srun  python ../scripts/alf.py ldss3_dr269_n1600_Re8_wave6e
