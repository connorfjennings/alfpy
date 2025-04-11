#!/bin/bash

#SBATCH --job-name=test_alf
#SBATCH --output=test_alf_job.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=20:00:00

export ALF_HOME="/home/cj535/alf/"
export ALFPY_HOME=/home/cj535/alfpy/
module load OpenMPI/4.1.4-intel-compilers-2022.2.1
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alfpy_env_py311
python -m pip install astro-sedpy

cd /home/cj535/alfpy/subjobs 
srun python ../scripts/alf_build_model.py GCcombine_r8vd275_001
srun python ../scripts/alf.py GCcombine_r8vd275_001
